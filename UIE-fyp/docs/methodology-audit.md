# Methodology audit — independent verification of committed artifacts

Audited commit `c77620e` ("Add files via upload") on branch `arena/01a0a56a-uie-fyp`.
Every number below was recomputed from the committed CSVs with
`numpy 2.4.6 / pandas 3.0.5 / scikit-learn 1.9.1 / scipy 1.17.1`.
No image data was required; `dataset/` is correctly git-ignored.

**Reproducibility check passed.** Refitting `RandomForestRegressor(200, min_samples_leaf=2,
random_state=42)` on the train split with the 10 final features reproduces the committed
`results/comparison/rf_baseline_test_predictions.csv` metrics *exactly*
(SSIM R² = 0.3552, PSNR R² = 0.2029). The committed feature stage is reproducible.

---

## CRITICAL

### C1. A pandas index-alignment bug scrambled every label in `ranking_combined.csv` — FIXED

**Status: root-caused, fixed, guarded by an assertion, and the whole downstream
chain re-run.** This invalidated the previously reported final feature set.

#### The symptom

`scripts/feature_ranking.py` writes three files from the same in-memory arrays in
one run, min-max normalising each target so its maximum raw importance becomes
exactly `1.000`. That invariant was violated in the committed artifacts:

| file | rank 1 feature |
|---|---|
| `ranking_ssim.csv` | `red_ratio` (raw importance 0.002086) |
| `ranking_psnr.csv` | `red_ratio` (raw importance 2.441547) |
| `ranking_combined.csv` | `edge_density` (`importance_ssim = 1.000`, `importance_psnr = 1.000`) |

`red_ratio` appeared at **combined rank 20** with `importance_ssim = 0.0325`, while
`edge_density` held `1.000` despite a raw value of 0.000063. A feature cannot be
both the maximum and near-minimum of the same normalisation.

#### The root cause

```python
combined = pd.DataFrame({
    "feature": FEATURE_NAMES_25,                                   # a plain list
    "importance_ssim": minmax(r_ssim.set_index("feature")[...]),   # a Series
    "importance_psnr": minmax(r_psnr.set_index("feature")[...]),   # a Series
})
```

The two Series are each sorted by their own target's importance, so they carry
**different index orders**. When a DataFrame is constructed from a dict mixing a
plain list with indexed Series, pandas takes the **sorted union** of the Series
indices as the frame index; the Series columns align **by index** (correct), but
the plain list is assigned **positionally** (wrong). Row *i* therefore received
the label `FEATURE_NAMES_25[i]` while holding the importances of
`sorted(FEATURE_NAMES_25)[i]`.

Both index orders are known, so the permutation is fully determined — and it
predicts the symptom exactly. `red_ratio` was the true maximum for both targets;
it sits at **alphabetical position 21**; `FEATURE_NAMES_25[21]` is
**`edge_density`**. That is precisely the corrupted row observed.

#### Proof

De-scrambling the committed file with `true_name = sorted(FEATURE_NAMES_25)[FEATURE_NAMES_25.index(label)]`
reconciles it with both per-target files *exactly*:

```
importance_ssim matches min-max(ranking_ssim) after de-scramble : True
importance_psnr matches min-max(ranking_psnr) after de-scramble : True
committed combined rank1 labelled = edge_density | de-scrambled = red_ratio
```

So `ranking_ssim.csv` and `ranking_psnr.csv` were **correct**; only
`ranking_combined.csv` was corrupted. (An earlier hypothesis in this document —
that the per-target files were stale — was wrong, and was disproved by adding
the assertion below and watching it fire on a *fresh* run.)

#### Blast radius

`feature_correlation.py` consumes `ranking_combined.csv` and walks it in rank
order, so the greedy redundancy filter kept and dropped the **wrong features**.
Everything downstream inherited the error: `selected_features.csv`,
`removed_redundant_features.csv`, `feature_subset_evaluation.csv`,
`final_selected_features.csv`, and the RF baseline test result. The chain was
*internally self-consistent with the corruption*, which is exactly why nothing
flagged it — each stage faithfully consumed the previous stage's output.

The previously reported final 10 was
`edge_density, rms_contrast, mean_blue, contrast, mean_saturation, mean_value,
homogeneity, mean_red, glcm_variance, dynamic_range`. The corrected true top-12
ranking is `red_ratio, mean_red, dynamic_range, entropy, correlation,
keypoint_density, gradient, homogeneity, mean_blue, edge_density, mean_value,
rms_contrast`. **Only five names appear in both lists**, and at different ranks.

#### The fix

Build the frame from the two Series alone so everything aligns by index, then
promote the index to a column:

```python
imp = pd.DataFrame({
    "importance_ssim": minmax(r_ssim.set_index("feature")["importance_mean"]),
    "importance_psnr": minmax(r_psnr.set_index("feature")["importance_mean"]),
})
imp.index.name = "feature"
combined = imp.reset_index()
assert sorted(combined["feature"]) == sorted(FEATURE_NAMES_25)
```

Plus a permanent guard at the end of the script, so a future refactor cannot
reintroduce it silently:

```python
for tag, r, col in (("ssim", r_ssim, "importance_ssim"),
                    ("psnr", r_psnr, "importance_psnr")):
    assert combined.loc[combined[col].idxmax(), "feature"] == r.iloc[0]["feature"], \
        f"INCONSISTENT RANKING ARTIFACTS ({tag}) - re-run this script"
```

The assertion fired on the unfixed code and passes on the fixed code.

#### Corrected results

Re-running ranking → correlation → subset evaluation gives **14 survivors**
(was 12) and a **flat** k-curve whose best value is k = 14, i.e. all survivors:

| k | SSIM R² (val) | PSNR R² (val) | avg R² |
|---|---|---|---|
| **14** | **0.5392** | **0.3674** | **0.4533** |
| 12 | 0.5220 | 0.3576 | 0.4398 |
| 10 | 0.5175 | 0.3585 | 0.4380 |
| 8 | 0.5222 | 0.3577 | 0.4400 |
| 6 | 0.5052 | 0.3320 | 0.4186 |

The 25-feature RF scores val avg R² = 0.4539 against 0.4533 for the 14, so the
**redundancy filter is free dimensionality reduction (25 → 14) but the top-k
truncation adds nothing**. Test, on the corrected 14 features (n = 133):

| target | R² | RMSE | MAE | Pearson r |
|---|---|---|---|---|
| SSIM | 0.3576 | 0.0890 | 0.0697 | 0.6065 |
| PSNR | 0.2249 | 2.7699 dB | 2.0895 dB | 0.4753 |
| avg (project-defined) | 0.2913 | — | — | — |

PSNR R² improves from 0.2029 (corrupted 10) to **0.2249** (corrected 14); SSIM
is essentially unchanged at 0.3576 vs 0.3552.

### C2. The CNN has never been run

Absent from the repository:

```
MISSING results/cnn/
MISSING results/comparison/baseline_metrics.csv
MISSING results/comparison/ablation_results.csv
MISSING models/
```

Only `results/comparison/rf_baseline_test_predictions.csv` exists. So:

- The proposed `HybridCNN` has **no results at all**.
- Baselines `mlp_final` and `image_only_nofeat` have no results.
- Ablation C (`hybrid_all25`) vs D (`hybrid_final`) — the experiment that answers the
  project's central research question, *"does feature selection improve CNN-based
  quality prediction?"* — has not been run.

Any claim that the CNN/fusion stage is validated is currently unsupported. This is the
largest remaining gap and it is on the critical path.

### C3. The headline R² values are validation-selection numbers, not test results

`results/feature/feature_subset_evaluation.csv` (scored on **val**, used to pick k):

| k | SSIM R² | PSNR R² | avg R² |
|---|---|---|---|
| 12 | 0.4850 | 0.3422 | 0.4136 |
| **10** | **0.5108** | **0.3595** | **0.4352** |
| 8 | 0.5065 | 0.3573 | 0.4319 |
| 6 | 0.4558 | 0.3661 | 0.4109 |

Actual **test** performance of the RF on those same 10 features (recomputed, matches the
committed predictions file):

| target | R² | RMSE | MAE | Pearson r |
|---|---|---|---|---|
| SSIM | **0.3552** | 0.0891 | 0.0691 | 0.6033 |
| PSNR | **0.2029** | 2.8090 dB | 2.1276 dB | 0.4525 |
| avg (project-defined) | **0.2791** | — | — | — |

The 0.511 / 0.360 / 0.435 triple is the *selection-time* estimate and is optimistically
biased by construction — k was chosen to maximise it on the same 134 validation images.
The defensible headline is **SSIM R² ≈ 0.36, PSNR R² ≈ 0.20, avg ≈ 0.28 on 133 test images.**

Bootstrap 95% CIs (4000 resamples, n = 133):

- SSIM R² ∈ **[0.150, 0.518]**
- PSNR R² ∈ **[0.065, 0.317]**

Report the test numbers with these intervals. Never present the val numbers as results.

---

## HIGH

### H1. k = 10 is not a statistically supported optimum

- k=10 beats k=8 by **0.0033** avg val R² on a single 134-image split — far inside noise.
- The per-target optima **disagree**: SSIM is best at k=10 (0.5108), PSNR is best at
  k=6 (0.3661 > 0.3595). k=10 wins only because the project-defined average favours it.
- Selecting k by a project-defined mean of two R² values, with a 0.003 margin, cannot be
  described as "k = 10 is the best subset".

**Fix:** repeated stratified K-fold (5 folds × 5 seeds) for the subset sweep, report mean
± SD per k, and either justify k=10 by parsimony/stability or pick k by paired test.
At minimum, present the k-curve as flat-within-noise rather than as an optimum.

### H2. All predictive value is joint — and the ranking IS meaningful once C1 is fixed

Two separate claims here, and C1 conflated them.

**(a) No single feature is individually predictive.** This is true and was never
an artifact. Fitting one RF per feature and scoring on test (n = 133):

| combined rank | feature | importance | SSIM R² alone | PSNR R² alone |
|---|---|---|---|---|
| 1 | `red_ratio` | 1.0000 | 0.0439 | −0.0170 |
| 2 | `mean_red` | 0.2878 | **0.0649** | −0.0164 |
| 3 | `dynamic_range` | 0.1775 | 0.0525 | −0.0634 |
| 4 | `entropy` | 0.1202 | −0.1235 | −0.1960 |
| 5 | `correlation` | 0.1054 | −0.0023 | −0.1755 |
| 6 | `keypoint_density` | 0.0851 | −0.2322 | −0.1545 |
| 10 | `edge_density` | 0.0432 | −0.0163 | 0.0166 |
| 11 | `mean_value` | 0.0369 | −0.0046 | 0.0540 |

The best standalone R² anywhere in the 25 is **0.065**, and most are negative.
Yet 14 features together reach test R² 0.358 (SSIM). So essentially **all**
predictive power is multivariate — the descriptors only work in combination.
That is a legitimate and interesting finding, and it should be stated this way
rather than narrated as "the ranking identifies the most informative quality
descriptors", which the standalone numbers do not support.

**(b) The ranking does track standalone predictive power.** With the correct
labels (post-C1):

| comparison | Spearman ρ | p |
|---|---|---|
| importance vs mean standalone R² | **+0.413** | **0.0403** |
| rank vs SSIM R² alone | +0.380 | 0.0613 |
| rank vs PSNR R² alone | +0.259 | 0.2105 |

Significant at α = 0.05 on the importance measure, marginal per-target. The top
three ranked features (`red_ratio`, `mean_red`, `dynamic_range`) are exactly the
three with positive standalone SSIM R². **Before the C1 fix these same
correlations were ρ = +0.087 (p = 0.678), +0.038 (p = 0.855) and +0.049
(p = 0.815)** — the label scrambling had destroyed the ranking's apparent
validity entirely. An earlier version of this document concluded the ranking was
meaningless; that conclusion was an artifact of C1 and is withdrawn.

The interpretation caveat that survives the fix: permutation importance measures
*conditional* contribution given the other 24 features, so it is not expected to
equal standalone power. Correlating at ρ ≈ 0.41 is reassuring, not definitional.

**Recommended addition:** ship the standalone-test-R² column next to the ranking
table. It is cheap (25 one-feature RFs, computed *after* final evaluation so the
selection protocol stays clean), it pre-empts the obvious viva question, and it
converts finding (a) from a weakness into a documented property of the descriptor.

### H3. No augmentation, no pretrained backbone, 623 training images

`cnn/dataset.py` applies **no augmentation whatsoever** — only letterbox → /255 → CHW.
`cnn/model.py` deliberately uses no ImageNet weights (documented rationale: keep the
image-only vs hybrid comparison attributable to the handcrafted features, not to ImageNet
priors — that reasoning is sound and worth keeping). But 623 images against a ~0.43 M
parameter CNN trained from scratch will overfit.

**Target-invariance analysis for augmentation** (this is the part that is easy to get
wrong here, because the labels are *metrics against a reference*). Rather than reason
about it, this was measured: all 25 features were recomputed on a transformed copy and
compared against the untransformed value. Results, as relative deviation:

| transform | 17 non-GLCM features | 8 GLCM features | verdict |
|---|---|---|---|
| horizontal flip | ≤ 2.2e−16 | **0.0 exactly** | safe |
| vertical flip | ≤ 1.6e−16 | **0.0 exactly** | safe |
| hflip + vflip | **0.0 exactly** | **0.0 exactly** | safe |
| 90° rotation | **0.0 exactly** | **up to 3.8e−02** | **UNSAFE** |

Both flips are exactly feature-invariant, including the GLCM block. The reason is that
`graycomatrix` uses `angles=[0]`, i.e. *horizontal* adjacency, and `symmetric=True`:
a horizontal flip reverses the neighbour order (absorbed by symmetrisation) and a
vertical flip leaves horizontal adjacency untouched. A 90° rotation converts horizontal
adjacency into vertical adjacency, so `contrast` (3.8%), `dissimilarity` (2.6%) and
`ASM`/`glcm_variance` (1.8%) all shift.

Targets are invariant under *any* of these, because SSIM and PSNR are computed between
preprocessed and reference and applying the same rigid transform to both leaves both
metrics unchanged.

- **Photometric augmentation (brightness/contrast/colour jitter, gamma) — ILLEGAL here.**
  It changes the preprocessed image without changing the reference, so the SSIM/PSNR
  targets genuinely change. Applying it without recomputing labels injects label noise
  directly into training.

**Recommendation:** augment with the Klein four-group {identity, hflip, vflip, hflip+vflip}
— a provably exact **4×** expansion of the 623 training images with zero relabelling and
zero feature recomputation. Rotations must stay out unless GLCM is switched to 4-angle
averaging (see H4), which would make them safe and give 8×.

### H4. GLCM uses a single orientation

`graycomatrix(gray, distances=[1], angles=[0], ...)` — one distance, one angle. Underwater
scenes have no canonical orientation, so 8 of the 25 descriptors are orientation-dependent
and the standard Haralick practice (average over 4 angles) is not followed. Averaging over
`[0, π/4, π/2, 3π/4]` would make them rotation-robust *and* unlock the augmentation in H3.
Cost: full feature re-extraction and a re-run of ranking → correlation → subset selection.
Decide before running the CNN, not after.

### H5. Single seed, single split — no uncertainty quantification anywhere

One 70/15/15 split, `RANDOM_STATE = 42`, one training run per model. With n_test = 133 the
bootstrap CIs are ±0.18 (SSIM) and ±0.13 (PSNR) wide. Ablation differences of 0.02–0.05
will be indistinguishable from noise, which means the central research question cannot be
answered by a single run.

**Fix:** repeated CV for selection (H1) plus ≥5 seeds for each neural model, and a paired
test (Wilcoxon on per-image squared errors, or a corrected resampled t-test) for
D vs C and D vs baselines. Report PSNR error in dB — an RMSE of 2.81 dB is the number an
examiner will care about, and R² hides it.

---

## MEDIUM

### M1. Negative permutation importances make "0.000" mean "actively harmful"

Raw importances contain negatives: **10 of 25** for SSIM (min −7.01e−05), **6 of 25** for
PSNR (min −0.0452). Negative = shuffling the feature *improved* held-out performance.
Min-max normalisation over a range that straddles zero maps the most-negative feature to
`0.000` and the max to `1.000`. So `glcm_entropy` (`importance_ssim = 0.000`) and
`correlation` (`importance_psnr = 0.000`) are the *most harmful* features for that target,
not merely the least important. State this wherever the normalised table appears, or the
0.000 entries will be misread.

### M2. Four of the 25 features are exact algebraic duplicates

Verified to machine precision on `feature_quality_dataset.csv`:

| identity | evidence |
|---|---|
| `std` ≡ `rms_contrast` | max abs diff = **0.0**, r = 1.0000000000 |
| `variance` = `std`² | max abs diff = **1.4e−12** |
| `ASM` = `energy`² | by construction in `features.py` |
| `ASM` ≡ affine(`glcm_variance`) | r = **1.0000000000** |

The last one is worth spelling out: with `normed=True` on a 256×256 GLCM,
`glcm_variance = var(p) = E[p²] − (1/65536)²` and `ASM = Σp² = 65536·E[p²]`, so
`ASM = 65536·glcm_variance + const`. They are the same dimension up to scale
(observed ratio 65585 → 77907, varying exactly as the constant term predicts).

**Consequence:** the "25-dimensional handcrafted descriptor" spans **21 independent
dimensions**. The correlation filter does remove all four redundancies correctly, so
nothing wrong reaches the final 10 — but the claim of 25 features invites the question,
and the answer should be pre-empted. Note also that `glcm_variance` surviving at rank 9
means the final-10's texture content is GLCM contrast + homogeneity + GLCM energy².

### M3. The committed validation report is stale

`results/feature/dataset_validation_report.csv` has `preprocessed_exists = True` for
**0 of 890** rows. The report was generated before `run_preprocessing.py`. Re-run
`scripts/validate_dataset.py` after preprocessing so the committed artifact reflects the
final state.

### M4. One README verification claim is not backed by any committed artifact

README §0.F claims "paired perceptual-hash distance median 2 vs **32 for mismatched
pairs**". The median-2 half is confirmed (median 2.0, mean 3.22, max 14). The
mismatched-pair control distribution is stored nowhere — the report has no such column.
Either persist the control (e.g. shuffled-pair Hamming distances) or drop the "vs 32".

Verified from the same report: 890 rows, `raw_exists` and `reference_exists` all True,
`raw_ref_dim_match` all True (**0** dimension mismatches), `raw_ref_identical_bytes` all
False. Zero NaNs and zero zero-variance columns in the feature matrix. SSIM range
[0.2442, 0.9664], PSNR range [9.69, 29.55] — both physically plausible.

### M5. Systematic range compression in the RF baseline

| | true SD | predicted SD | true range | predicted range |
|---|---|---|---|---|
| SSIM | 0.1114 | 0.0695 | [0.411, 0.932] | [0.457, 0.866] |
| PSNR | 3.158 dB | 1.554 dB | [10.98, 28.82] | [11.94, 19.81] |

The model **cannot predict a PSNR above ~19.8 dB** although the test set reaches 28.8 dB.
Classic RF regression-to-the-mean, but it means the predictor is weakest exactly where a
quality estimate is most useful (identifying genuinely good enhancements). The CNN may
behave differently; report predicted-vs-true range and a calibration/slope check for every
model, not just R².

---

## LOW — documentation defects

| # | Location | Problem |
|---|---|---|
| L1 | `src/config.py` line ~40 | **FIXED.** Comment said `-> 623 / 133 / 134 images`; actual split is train 623, **val 134, test 133** |
| L2 | `scripts/make_split.py` docstring | **FIXED.** Same swapped `623 / 133 / 134` |
| L3 | `README.md` §0.G | **FIXED.** "Option A (MLP on **8** features)" — the final set is 10 |
| L4 | `scripts/run_ablation.py` docstring | **FIXED.** "B: **8** handcrafted features only" — B reuses `baseline_rf`, which uses the final **10** |
| L5 | `README.md` §10 | **FIXED.** Advertises `Top-k ∈ {15, 12, 10, 8, 6}`; k=15 is unreachable (only 12 survivors). The script clips correctly and prints it — the README does not mention the clip |
| L6 | `cnn/train.py::set_seeds` | **FIXED (removed).** `torch.set_num_threads(max(1, torch.get_num_threads()))` is a no-op |
| L7 | `scripts/run_baselines.py::rf_baseline` | **FIXED.** Fits each RF twice (once for metrics, once for the predictions CSV). Deterministic, so no correctness impact — just 2× cost |
| L8 | `README.md` | **Physically truncated mid-word.** The committed file ended at §11 with `...keypo` followed by the literal text `...[truncated 4045 chars]` — 8368 bytes total. §11–§19 were missing, including the Limitations section that `validate_dataset.py` explicitly tells the reader to consult ("See README section 19 (Limitations)"). Reconstructed from the code and results |

L3/L4 are the same stale-"8" confusion the project set out to eliminate; they are the
first things a reader will notice.

---

## Concerns I raised earlier that the code DISPROVES

Recorded so the audit is not one-sided.

**D1. Two-head loss scaling — already handled correctly.** `cnn/dataset.py` fits a
`StandardScaler` on `[ssim, psnr]` using **train rows only**, and `cnn/evaluate.py`
calls `inverse_transform` before computing any metric. The MSE-on-normalised-targets
justification is stated in `train.py`. My concern that PSNR (10–40 dB) would swamp
SSIM (0–1) does not apply. ✓

**D2. Variable-resolution feature bias — not a real problem.** Heights do vary
enormously (140 distinct values, 266 → 901 px, a **3.39×** pixel-count range), but:

- `pearson(out_h, ssim) = −0.155`, `pearson(out_h, psnr) = −0.094`
- `|pearson(out_h, f)| ≤ 0.074` for *every* scale-sensitive feature
  (`edge_density` −0.017, `gradient` −0.048, `laplacian_variance` −0.074,
  `keypoint_density` +0.000, `contrast` −0.063, `homogeneity` −0.040,
  `glcm_variance` −0.032)

Because *all* images are resized to the same width, the downsample factor is uniform, so
feature scale stays comparable. My earlier worry was overstated. Residual risk is only
estimator noise at low pixel counts (one image has 159 k px). Worth one sentence in
Limitations; not worth a code change.

**D3. Redundancy surviving into the final 10 — the filter worked.** Highest surviving
pair is `mean_saturation ~ mean_value` at **r = 0.832**, then `edge_density ~ contrast`
0.827, `edge_density ~ homogeneity` −0.815, `rms_contrast ~ dynamic_range` 0.808.
The three "contrast-flavoured" survivors are *not* mutually redundant:
`rms_contrast ~ contrast` = 0.391, `contrast ~ dynamic_range` = 0.336. All below the
documented 0.90 threshold. 19 of 300 pairs in the full 25 exceed 0.90. ✓
Optional strengthening: sweep the threshold (0.80 / 0.85 / 0.90 / 0.95) to show 0.90 is
not arbitrary — four pairs sit in the 0.81–0.83 band just under it.

**D4. Group-aware duplicate split — verified.** All 7 byte-identical duplicate pairs are
co-located, every one in `train`, **0 violations**. ✓

**D5. Leakage protocol — verified in code and sound.** Ranking fits on train and
permutates on val (`feature_ranking.py`), correlation matrix on train
(`feature_correlation.py`), subset sweep scored on val with test never read
(`feature_subset_evaluation.py`), both feature and target scalers fit on train only
(`cnn/dataset.py::fit_scalers`), test touched once in `evaluate.py` /
`run_baselines.py`. Preprocessing is entirely per-image (gray-world gains, CLAHE,
bilateral, `γ = clip(mean/128, 0.5, 2.0)`) — no cross-image statistic is fitted, so no
preprocessing leakage. Reference images appear only in target columns.
This is genuinely better than most published undergraduate pipelines. ✓

---

## Status of the recommended work

| # | item | status |
|---|---|---|
| 1 | Re-run `feature_ranking.py`, add the consistency assertion (C1) | **DONE** — bug found and fixed, assertion added, full downstream chain re-run; final set is now 14 features |
| 2 | Decide GLCM angles before training (H4) | **OPEN — decision needed.** Left at `angles=[0]`, so rotations stay excluded from augmentation |
| 3 | Add flip augmentation (H3) | **DONE** — Klein four-group, train-only, invariance measured not assumed |
| 4 | Run the CNN suite (C2) | **IN PROGRESS** — RF and MLP complete, image-only / hybrid / hybrid-all25 training |
| 5 | Repeated CV for the subset sweep (H1) | **OPEN** — the k-curve is documented as flat; repeated CV would let k be chosen defensibly |
| 6 | Per-feature standalone test R² (H2) | **DONE as analysis**, recorded above; not yet a committed artifact |
| 7 | Doc defects L1–L7, validation staleness M3, phash control M4 | **L1–L8 DONE**, M4 code **DONE** (control now persisted; re-run `validate_dataset.py` to write it), M3 **OPEN** (needs a validation re-run now that `dataset/preprocessed/` exists) |
| 8 | Report test numbers with bootstrap CIs and PSNR in dB (C3, H5, M5) | **DONE** — CIs wired into `evaluate.py` and `run_baselines.py` |
| — | Multi-seed runs (H5) | **WIRED, not yet run** — `CNN_SEEDS = (42, 43, 44)` and `--seed`; the suite currently runs seed 42 only |

### Two open decisions that need a human

1. **H4 — GLCM angles.** Switching to `angles=[0, π/4, π/2, 3π/4]` averaged makes
   the 8 texture descriptors orientation-robust and unlocks 8× augmentation
   instead of 4×. It forces full feature re-extraction (~3 min) and a re-run of
   ranking → correlation → subset selection (~1 min), and it invalidates every
   CNN checkpoint trained so far. Cheap now, expensive after the suite finishes.
2. **M6 — Spearman redundancy filter.** Pearson misses deterministic nonlinear
   duplicates: `variance = std²` survives the current filter at rank 14 because
   `std` was already dropped and Pearson(variance, entropy) < 0.90, even though
   Spearman(variance, std) = 1.0 exactly. Adding a Spearman pass (or replacing
   Pearson) would remove it and give 13 survivors. This is a defensible
   methodological strengthening, not a bug fix, so it is a choice rather than a
   correction.

### Reporting discipline

`run_baselines.py` already prints the right rule — *"if hybrid_final does not
beat the baselines, that is the finding"* — and it should be honoured. The
evidence so far points that way: the RF reaches test R² 0.3576 / 0.2249 while the
MLP on the same 14 features reaches only 0.3255 / 0.1407, i.e. **a 3 106-parameter
MLP is currently losing to a random forest**, and the bootstrap CI on the MLP's
PSNR R² includes zero ([−0.001, 0.262]). Given that no single feature explains
anything on its own and the k-curve is flat, a null or negative result for the
hybrid is a live possibility. The project is methodologically strong enough that
reporting it honestly is a better outcome than a tuned-up number that will not
survive questioning.
