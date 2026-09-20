# Underwater Image Quality Analysis with Handcrafted Features and CNN (UIEB)

Final-year project: predict how close a **classical enhancement pipeline's**
output is to the **UIEB reference image**, using (a) 25 handcrafted image
descriptors with ranking + redundancy filtering, and (b) a **hybrid CNN +
selected-features** regressor. All results below are produced by running the
code in this repository — nothing is hand-written or fabricated.

> **Task definition (read this first).** The model does NOT restore images.
> It performs **multi-output regression**: from the *preprocessed* image (+ its
> handcrafted features) it predicts **SSIM** and **PSNR** measured against the
> UIEB *reference* image. The reference is a **target source only**, never a
> model input. Equivalently: this is a **no-reference (blind) quality estimator
> trained to reproduce full-reference metrics**, because in deployment no
> reference image exists.

**Document map**

| document | purpose |
|---|---|
| `README.md` (this file) | the authoritative report — method, results, defects, limitations |
| `docs/project-flow.md` | **the roadmap**: 9 phases with validation gates, the leakage firewall, ranked upgrade list, thesis chapter mapping, viva pack |
| `docs/architecture.md` | per-file explanation of every module and why it exists |
| `docs/plain-language-explanation.md` | the whole project with no assumed background |
| `docs/methodology-audit.md` | the independent audit: C1 label-scrambling bug, the two feature bugs, H1–H5, L1–L8 and their status |

---

## 0. Method validity check (required by the project brief)

### A. What is correct
- Predicting SSIM/PSNR of (preprocessed, reference) from preprocessed-derived
  descriptors is a coherent supervised task (full-reference quality *prediction*).
- Random-Forest + **permutation importance on held-out data** is a sound
  ranking method; Pearson `|r| ≥ 0.90` greedy redundancy removal (rank order,
  keep higher-ranked) is sound.
- R²/RMSE/MAE per target are the right regression metrics; "accuracy" is
  meaningless here and is never reported.
- The classical preprocessing (per-image white balance, CLAHE, bilateral,
  per-image gamma) fits no cross-image statistics → no leakage at that stage.

### B. What is questionable
- The **combined R²** (mean of SSIM-R² and PSNR-R²) is a project-defined
  summary, not a standard metric — always labelled as such.
- `glcm_variance` (variance of GLCM entries) is a project-defined descriptor,
  not the classic Haralick texture variance — kept under a documented definition.
- PSNR is much harder to predict from global descriptors than SSIM (pixel-level
  vs structural) — expect lower PSNR-R²; that is normal, not a bug.

### C. What had to be changed (with reasons)
1. **uint8 overflow bug in `colorfulness`** (critical): the draft computed
   `R − G` / `R + G` on uint8 arrays, where subtraction underflows
   (`10 − 200 → 66`) and addition overflows (`200 + 100 → 44`). Channels are
   now cast to float32 first. The *definition* is unchanged.
2. **BGR/RGB channel order** (critical): OpenCV reads BGR but the feature code
   assumes RGB. Every caller now converts explicitly; otherwise `mean_red` /
   `mean_blue` / `red_ratio` would be silently swapped.
3. **Leakage-free protocol**: ranking/correlation/subset selection now use
   **train (+ val) only**; test is touched once for final reporting. Earlier
   all-890 selection results are exploratory and potentially optimistic.
4. **UIEB duplicate groups**: 7 byte-identical raw pairs exist (4 with genuinely
   different references). The split is **group-aware** (duplicates share one
   split) instead of silently dropping data.
5. **Documented ambiguities**: CLAHE channel (L of CIELAB), adaptive-gamma
   formula `γ = clip(mean/128, 0.5, 2.0)`, CNN input letterboxing — all fixed
   and documented because the original notes gave only parameter values.

### D. What can remain
- The 25 feature *definitions*, RF hyperparameters, permutation protocol
  (`n_repeats=5`, `neg_mean_squared_error`), `|r| ≥ 0.90` threshold, subset
  grid {15, 12, 10, 8, 6}, hybrid fusion concept — all retained.

### E. Data-leakage risks (all mitigated)
| Risk | Mitigation |
|---|---|
| Selection on all 890, test from same pool | Selection on train/val only; test once |
| Scaler fit on full data | `StandardScaler` fit on train only, saved |
| Raw duplicates across splits | Group-aware split by content hash |
| Reference as input | References only generate target columns |
| Preprocessing fitting dataset stats | All steps are per-image |

### F. Reference-image risks
- UIEB references are **human-preferred pseudo-references** (best-of-several
  algorithm outputs picked by volunteers — Li et al. TIP 2019), not physical
  ground truth. SSIM/PSNR therefore measure *similarity to a preferred
  enhancement*. Pairing genuineness is **verified**, not assumed: 0 dimension
  mismatches, paired perceptual-hash distance median 2 vs 32 for mismatched
  pairs, plus visual contact sheets in `plots/`.

### G. CNN architecture decision
- **Option A (MLP on the final selected features)** is a baseline, and is
  called an MLP — never a CNN. (An earlier revision said "8 features"; the
  selected-set size is whatever §10 chooses, not a fixed constant.)
- **Option B (image-only CNN)** is a baseline/ablation.
- **Option C (hybrid)** is the proposed model: lightweight image branch
  (4 conv blocks + GAP) + feature branch, concatenated, one head with 2
  normalised outputs. Scientifically sensible *iff* baselines show fusion
  helps — reported honestly either way.

### Final approved pipeline
`UIEB raw → classical preprocessing → {25 features → ranking → correlation
filter → subset eval → final-k} + {letterboxed image → CNN} → fusion →
SSIM + PSNR`, with reference images as target sources only, and a fixed
group-aware 70/15/15 split.

---

## 1. Dataset
UIEB (Li et al., *"An Underwater Image Enhancement Benchmark Dataset and
Beyond"*, TIP 2019): 890 raw underwater images + 890 paired reference images
(+ 60 reference-less challenging images, not used for supervised targets).
Official source: <https://li-chongyi.github.io/proj_benchmark.html> (Google
Drive IDs `12W_kkblc2Vryb9zHQ6BfGQ_NKUfXYk13` raw / `1cA-8CzajnVEL4feBRKdBxjEe6hwql6Z7`
reference). Downloaded here via the GitHub mirror `JJsnowx/UIEB_Dataset`
(verified: 890+890 PNGs, identical contiguous names `UIEB_0..UIEB_889`).
`dataset/` contents are git-ignored (~1.6 GB) and reproducible via
`scripts/download_uieb.py`.

## 2. Preprocessing
`src/preprocess.py`: aspect-preserving resize (width 600, INTER_AREA) →
Gray-World white balance → CLAHE on CIELAB L-channel (2.5, 8×8) → bilateral
(9, 75, 75) → adaptive gamma `clip(mean/128, 0.5, 2.0)`. Reference images get
the *identical* geometric resize before SSIM/PSNR. Runs: `run_preprocessing.py`.

## 3. The 25 handcrafted features
Statistical: `mean, std, variance, entropy, dynamic_range, rms_contrast`.
Colour: `mean_red/green/blue, colorfulness, red_ratio, mean_saturation,
mean_value`. Texture: `contrast, correlation, energy, homogeneity, ASM,
dissimilarity, glcm_entropy, glcm_variance`. Edge/sharpness: `edge_density,
gradient, laplacian_variance, keypoint_density`. Glossary: Canny = strong
intensity discontinuities → binary map; `edge_density` = edge-pixel fraction;
SIFT = scale/rotation-robust local keypoints, density = count/area; GLCM =
gray-level co-occurrence → texture; `energy` = uniformity, `ASM = energy²`;
`entropy` = distributional complexity; Laplacian variance = sharpness-related.
No single feature is claimed to *be* quality — they are descriptors.

## 4. Feature extraction
`src/features.py::extract_features` (RGB uint8 in; float-safe colour maths;
single shared SIFT instance; NaN-guarded). Extracted from **preprocessed**
images only. Builder: `scripts/build_feature_dataset.py`.

## 5. SSIM / PSNR targets
skimage `structural_similarity(..., channel_axis=2, data_range=255)` and
`peak_signal_noise_ratio`. Shapes match by the shared resize (§2).

## 6. Random-Forest ranking
`RandomForestRegressor(200, min_samples_leaf=2, seed 42)` per target, fit on
**train** (623 imgs). Val R²/RMSE/MAE printed.

## 7. Permutation importance
`permutation_importance` on **val** (held-out), `n_repeats=5`,
`scoring=neg_MSE`. Shuffling-based held-out measure — distinct from impurity/MDI
importance. Per-target ranks min-max normalised and averaged → combined rank.

## 8. Pearson correlation
25×25 Pearson matrix on **train** features; pairs with `|r| ≥ 0.90` listed
(sign included; negative redundancy counts via absolute value).

## 9. Redundancy removal
Greedy walk in rank order; drop a feature iff `|r| ≥ 0.90` with a kept
higher-ranked feature. Target correlation is never a removal criterion.

## 10. Subset evaluation
Top-k ∈ {15, 12, 10, 8, 6} of survivors → RF per target on train, scored on
**val** (R²/RMSE/MAE + project-defined avg R²). Best-k → `final_selected_features.csv`.

Grid values above the survivor count are **clipped** to it and de-duplicated, so
the evaluated grid is `min(k, n_survivors)`. With 14 survivors the k=15 entry
evaluates as k=14; the script prints the clipped grid it actually used.

## 11. Final selected features

> **CORRECTION (supersedes everything previously reported here).** The committed
> `ranking_combined.csv` was produced by a pandas index-alignment bug in
> `scripts/feature_ranking.py`: the frame was built from a plain `feature` list
> *alongside* two Series indexed by feature name, so pandas indexed the frame by
> the sorted union of those indices, aligned the importance columns **by index**
> but assigned the `feature` list **positionally**. Every label was shifted by
> the permutation between canonical and alphabetical order — which is why the
> file claimed `edge_density` = 1.000/1.000 while both per-target files ranked
> `red_ratio` first. The per-target files were correct; the combined file was
> not. Fixed, guarded by an assertion, and the whole downstream chain re-run.
> See `docs/methodology-audit.md` §C1.

Determined by the leakage-free run in `results/feature/`. The historically
reported sets (an exploratory 8, then a 10) are **both obsolete**: the 8 came
from selection on all 890 images, and the 10 came from the scrambled ranking
above. Do not cite either.

**Corrected combined ranking (top 12 of 25):** `red_ratio`, `mean_red`,
`dynamic_range`, `entropy`, `correlation`, `keypoint_density`, `gradient`,
`homogeneity`, `mean_blue`, `edge_density`, `mean_value`, `rms_contrast`.

**Redundancy removal (|r| ≥ 0.90, greedy in rank order) drops 11 → 14 survivors:**

| removed | kept instead | r |
|---|---|---|
| `mean_red` | `red_ratio` | +0.9357 |
| `gradient` | `keypoint_density` | +0.9294 |
| `edge_density` | `keypoint_density` | +0.9365 |
| `rms_contrast` | `entropy` | +0.9153 |
| `glcm_entropy` | `homogeneity` | −0.9373 |
| `mean_green` | `mean_blue` | +0.9825 |
| `contrast` | `dissimilarity` | +0.9495 |
| `std` | `entropy` | +0.9153 |
| `colorfulness` | `mean_saturation` | +0.9668 |
| `ASM` | `glcm_variance` | +1.0000 |
| `energy` | `glcm_variance` | +0.9362 |

**Final 14:** `red_ratio, dynamic_range, entropy, correlation, keypoint_density,
homogeneity, mean_blue, mean_value, mean, dissimilarity, laplacian_variance,
glcm_variance, mean_saturation, variance`.

**Subset sweep (RF fit on train, scored on val):**

| k | SSIM R² | PSNR R² | avg R² (project-defined) |
|---|---|---|---|
| **14** | **0.5392** | **0.3674** | **0.4533** |
| 12 | 0.5220 | 0.3576 | 0.4398 |
| 10 | 0.5175 | 0.3585 | 0.4380 |
| 8 | 0.5222 | 0.3577 | 0.4400 |
| 6 | 0.5052 | 0.3320 | 0.4186 |

**Read this curve honestly: it is flat.** Best k = 14, i.e. *all* survivors, and
the spread from k=6 to k=14 is 0.035 avg R² on a single 134-image validation
split. The measured value of the selection stage is therefore in the
**redundancy filter (25 → 14 at no cost in accuracy** — the 25-feature RF scores
val avg R² 0.4539 vs 0.4533 for the 14**)**, *not* in the top-k truncation,
which adds nothing. Claiming an optimal k would not survive a significance test.

**Exact algebraic redundancies inside the 25** (verified to machine precision,
and the reason a 25-feature descriptor really spans 21 independent dimensions):
`std ≡ rms_contrast` (max abs diff 0.0), `variance = std²` (diff 1.4e−12),
`ASM = energy²`, and `ASM` is *affine-equivalent* to `glcm_variance`
(r = 1.0000000000) because with a normalised 256×256 GLCM,
`ASM = 65536·glcm_variance + const`. Note that `variance` still survives the
Pearson filter: Pearson is blind to the *quadratic* relation `variance = std²`
once `std` itself has already been dropped. A Spearman-based filter would catch
it (ρ = 1.0 exactly) — see §19.

## 12. Test-set results (RF baseline on the final 14 features)

Test is read exactly once. n_test = 133.

| target | R² | RMSE | MAE | Pearson r |
|---|---|---|---|---|
| SSIM | 0.3576 | 0.0890 | 0.0697 | 0.6065 |
| PSNR | **0.2249** | **2.7699 dB** | **2.0895 dB** | 0.4753 |
| avg R² (project-defined) | 0.2913 | — | — | — |

The gap from the val numbers in §11 (0.5392 / 0.3674) is expected and is *not*
a bug: k was chosen to maximise val performance, so val is optimistically
biased by construction. **Report the test numbers as the result.** Bootstrap
95% CIs are written into `baseline_metrics.csv` (`*_r2_ci95_lo/hi`); at n=133
they are roughly ±0.15 on R², so any ablation difference smaller than that is
not a finding.

Also report the **prediction range**, not just R²: the forest compresses it
(predicted PSNR spans ~12–20 dB against a true span of ~11–29 dB), so it is
weakest exactly where a quality estimate is most useful.

## 13. Baselines and the proposed model

`scripts/run_baselines.py` → `results/comparison/baseline_metrics.csv`:

| model | input | role |
|---|---|---|
| `baseline_rf` | final 14 features | classical-ML reference |
| `mlp_final` | final 14 features | neural, features only — an **MLP**, never called a CNN |
| `image_only_nofeat` | preprocessed image | neural, image only |
| `hybrid_final` | image **+** final 14 features | **proposed model** |

All four are evaluated in one joint pass over the untouched test split. The RF is
fit once per target and reused for both metrics and the predictions file.

## 14. Ablation — does feature selection help the CNN?

`scripts/run_ablation.py` → `results/comparison/ablation_results.csv`:

| | model | features |
|---|---|---|
| A | `image_only_nofeat` | none |
| B | `baseline_rf` | final selected |
| C | `hybrid_all25` | all 25 |
| D | `hybrid_final` | final selected |

**C vs D is the direct test of the feature-selection contribution**; D vs A tests
whether fusion helps at all. A, B and D are reused from §13 rather than
retrained, so the comparison is exact. Only C is trained by this script.

## 15. Hybrid CNN architecture

`cnn/model.py` — **late (feature-level) fusion**, ~0.43 M parameters:

```
preprocessed image ─ letterbox 224×224 ─ 4×ConvBlock(3→32→64→128→256)
                                          └ GlobalAvgPool → 256-d ┐
                                                                  ├─ concat 288-d
final 14 features ─ StandardScaler(train) ─ Linear→ReLU→Dropout ──┘   │
                                                    → 32-d ───────────┘
                                                                  ↓
                             Linear(288→128) → ReLU → Dropout(0.3) → Linear(128→2)
                                                                  ↓
                                                    [SSIM_norm, PSNR_norm]
```

`ConvBlock` = Conv3×3 → BatchNorm → ReLU → MaxPool2. **Global average pooling**
replaces Flatten+Dense: at 14×14×256 a flatten would feed ~50 k inputs into the
head and dominate the parameter count, which is unaffordable at 623 training
images. **No pretrained weights, deliberately** — ImageNet priors would confound
the image-only vs hybrid comparison, since a gain could come from pretraining
rather than from the handcrafted features.

## 16. Training protocol

`cnn/train.py`, identical for every neural model so comparisons are fair:

- Adam(lr 1e−3, weight decay 1e−4); **MSE on standardised targets**. A second
  `StandardScaler`, fit on train only, standardises `[ssim, psnr]` so that one
  loss treats both outputs fairly — SSIM ∈ [0,1] against PSNR ∈ ~[10,40] dB
  would otherwise be dominated by PSNR by orders of magnitude. Predictions are
  inverse-transformed before any metric is reported, so every published number
  is in original units (PSNR in dB).
- Early stopping on val loss, patience 12, max 80 epochs, **best** checkpoint
  kept (not last).
- Augmentation, **train split only**: a random element of the Klein four-group
  {identity, hflip, vflip, hflip+vflip}. These four were *measured* to leave all
  25 features invariant to within 2.2e−16 and to leave SSIM/PSNR invariant
  (the same rigid transform on both members of a pair changes neither metric),
  so the cached features and targets stay exactly correct — a free 4× expansion
  with no relabelling. **Rotations are excluded**: a 90° turn converts the
  GLCM's horizontal adjacency (`angles=[0]`) into vertical adjacency and shifts
  the 8 GLCM descriptors by up to 3.8%. **Photometric augmentation is excluded
  outright**: it changes the preprocessed image without changing the reference,
  so the targets genuinely change and the labels become wrong.
- Val/test are never augmented, so their metric is a fixed function of the split
  rather than of a random draw.
- `CNN_SEEDS = (42, 43, 44)`; non-default seeds append `_s<seed>` to the run tag
  so repeated runs never overwrite each other.

## 17. Reproducing everything

```bash
pip install -r requirements.txt
python scripts/download_uieb.py          # 890 raw + 890 reference + 60 challenging
python scripts/run_preprocessing.py      # -> dataset/preprocessed/, preprocessing_log.csv
python scripts/validate_dataset.py       # -> validation report, duplicate groups, contact sheets
python scripts/build_feature_dataset.py  # -> 25 features + SSIM/PSNR targets (CHECKS 3-10)
python scripts/make_split.py             # -> group-aware 623/134/133 (CHECK 11)
python scripts/feature_ranking.py        # -> ranking_{ssim,psnr,combined}.csv (+ C1 assertion)
python scripts/feature_correlation.py    # -> correlation matrix, 14 survivors
python scripts/feature_subset_evaluation.py  # -> k sweep, final_selected_features.csv
python scripts/run_baselines.py          # TEST read once -> baseline_metrics.csv
python scripts/run_ablation.py           # -> ablation_results.csv
python scripts/make_plots.py             # -> plots/
```

Each step above has a **validation gate** that must pass before the next one is
allowed to consume its output — the gates, the leakage firewall (which split each
stage may see), and the correct recovery order after a fresh clone are set out in
`docs/project-flow.md`. In particular: `dataset/` is gitignored, so a fresh clone
must re-run `download_uieb.py` → `run_preprocessing.py` →
`build_feature_dataset.py` before any later stage, and
`feature_quality_dataset.csv` must come out **bit-identical** to the committed
copy — if it does, Phases 4–5 remain valid and need not be re-run.

Verified end-to-end on Python 3.11 with numpy 2.4.6 / pandas 3.0.5 /
scikit-learn 1.9.1 / OpenCV 5.0.0 / scikit-image 0.26.0: preprocessing
reproduces `preprocessing_log.csv` exactly (heights 266–901, all widths 600,
γ ∈ [0.500, 1.412], mean 0.865), and feature extraction reproduces all
**27 columns × 890 rows of `feature_quality_dataset.csv` bit-for-bit**
(max relative drift 0.0e+00) despite the differing library versions.

## 18. Known defects and their status

`docs/methodology-audit.md` is the full independent audit. Summary:

| id | defect | status |
|---|---|---|
| C1 | pandas index-alignment bug scrambled every label in `ranking_combined.csv`, invalidating the previous final-feature set | **fixed**, assertion added, chain re-run |
| C2 | CNN never run — no `results/cnn/`, no baseline/ablation CSVs | in progress |
| C3 | val-selection R² presented as results | **fixed** — §12 reports test |
| H1 | k chosen on a 0.003 margin | documented; curve is flat, k=14 |
| H2 | ranking seemed uncorrelated with standalone power | **was a C1 artifact** — with correct labels ρ = +0.413, p = 0.040 |
| H3 | no augmentation | **fixed** — flip four-group, train only |
| H5 | single seed, no CIs | CIs added; multi-seed wired via `CNN_SEEDS` |
| L1–L7 | documentation defects | **fixed** |
| L8 | this README was truncated mid-word at §11, losing ~4 000 characters including the Limitations section that `validate_dataset.py` points readers to | **fixed** — §11–§19 reconstructed from the code and results |

## 19. Limitations

1. **The references are not ground truth.** UIEB's 890 references are
   *human-preferred pseudo-references*: the best of several existing enhancement
   algorithms' outputs, chosen by volunteer pairwise voting (Li et al., TIP 2019).
   SSIM/PSNR here therefore measure *similarity to a preferred enhancement*, not
   physical fidelity, and can penalise a restoration that is objectively better
   but stylistically different. The 60 challenging images have no reference and
   are excluded from all supervised targets.
2. **The model does not enhance anything.** It predicts the quality a fixed,
   deterministic classical pipeline attains. The contribution is no-reference
   prediction of full-reference metrics, not restoration.
3. **Modest explanatory power.** Test R² ≈ 0.36 (SSIM) and ≈ 0.22 (PSNR), with
   PSNR RMSE ≈ 2.77 dB. Bootstrap CIs are wide at n_test = 133.
4. **A non-canonical split.** UIEB ships an official 800/90 split; the 70/15/15
   split here is chosen to make leakage-free feature selection possible, at the
   cost of comparability with published UIEB numbers (WaterNet, FUnIE-GAN, UGAN).
5. **GLCM uses one orientation.** `distances=[1], angles=[0]` makes 8 descriptors
   orientation-dependent although underwater scenes have no canonical
   orientation. Averaging over `[0, π/4, π/2, 3π/4]` would be more robust and
   would additionally make 90° rotations augmentation-safe (4× → 8×).
6. **Pearson-only redundancy filtering.** It misses deterministic *nonlinear*
   dependencies: `variance = std²` survives because Pearson(variance, entropy)
   falls below 0.90 after `std` was already dropped, even though
   Spearman(variance, std) = 1.0 exactly. A Spearman (or mutual-information)
   filter would remove it.
7. **Small-data caveat.** 623 training images against a from-scratch CNN with no
   pretrained backbone. The flip augmentation and GAP-based branch are
   mitigations, not a substitute for data.
