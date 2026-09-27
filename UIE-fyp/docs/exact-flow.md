# The exact flow of this project

*Written 2026-09-20. This is a **description of what the code actually does**, in
exact order, with the artefacts each stage writes and the gate it must pass —
re-derived by reading every module and every committed result file, not copied
from `docs/project-flow.md`.*

> **How this differs from `docs/project-flow.md`.**
> `project-flow.md` is the **roadmap** (what to do, in what order, and what is
> still missing). This file is the **as-built blueprint**: the two flows that
> exist, the exact commands, the exact files, the numbers that came out, and —
> in §12 — every place where the documentation, the code and the committed
> artefacts disagree with each other.
>
> Nothing here is estimated. Every number was read out of
> `results/**/*.csv|json` or `git`, and every claim about behaviour carries a
> `file:line` anchor. Where something does **not** exist, it says so.

---

## 0. The whole project on one screen

**It is not one pipeline. It is two coupled pipelines that share the same base.**

```
                    ┌──────────────────── SHARED BASE (once) ────────────────────┐
UIEB raw 890 ──► P1 download/validate ──► P2 group-aware split 623/134/133 ──► 3a classical
                 │  (SHA-256 dup groups, phash control)        (frozen, seed 42)   enhancement
                 └───────────────────────────────────────────────────────────────┘
                                        │
              ┌─────────────────────────┴──────────────────────────┐
              │                                                    │
   TRACK A — ASSESSMENT (the FYP's core)              TRACK B — RESTORATION (Phase 3b)
   "how good is the classical output?"                "can a learned model do better?"
              │                                                    │
   preprocessed + reference → 25 features + SSIM/PSNR   preprocessed → U-Net → enhanced PNG
   → train-only ranking → train-only redundancy filter  (target = aligned reference)
   → k-sweep on val → final 14 features                 → train 623 pairs, val 134 full-res
   → RF / MLP / image-only CNN / HYBRID CNN             → best val-SSIM checkpoint (epoch 55)
   → sealed test read once → R²/RMSE/MAE + CIs          → 133 enhanced PNGs on disk
   → ablation A/B/C/D → does selection help?            → raw vs classical vs U-Net, paired tests
```

The two tracks are **currently disconnected in code**: Track A was trained only
on *classical* outputs and has never scored a *U-Net* output (that is upgrade U6).
That single sentence is the most important structural fact about this repository.

---

## 1. What is computed, precisely

| question | answer | where |
|---|---|---|
| Model input | the **preprocessed** (=classically enhanced) image, and optionally its 25 handcrafted features | `cnn/dataset.py:114-133` |
| Model output | two scalars: `SSIM`, `PSNR` — normalised during training, inverse-transformed before reporting | `cnn/evaluate.py:63-64` |
| Supervision | SSIM/PSNR of (preprocessed image, **UIEB reference**) — the reference is *never* an input | `scripts/build_feature_dataset.py:45-60` |
| So the task is | **no-reference (blind) prediction of full-reference metrics** | README §0 |
| Track B input/output | preprocessed image → **enhanced image of identical shape**, target = aligned reference | `cnn/unet.py:82-87` |
| Metric definition (single source) | `src/iqa.py::compute_ssim_psnr` — skimage, `channel_axis=2, data_range=255`, uint8; PSNR called as `(ref, test)` | `src/iqa.py:29-45` |

Split sizes, verified from the committed split file: **train 623 / val 134 /
test 133** (890 total, `results/feature/data_split.csv`).

Label population (all 890 classical pairs): SSIM `0.7710 ± 0.0991`
[0.2442, 0.9664]; PSNR `17.021 ± 2.872` dB [9.69, 29.55].
Test split (133): SSIM [0.4107, 0.9323], PSNR [10.98, 28.82] dB.

---

## 2. Phase 0 — environment

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # torch >=2.0, opencv-python-headless, sklearn, skimage, scipy, matplotlib, tqdm, torch, gdown
```

* **Gate:** imports succeed; versions recorded. Verified in a previous session on
  Python 3.11 / numpy 2.4.6 / pandas 3.0.5 / scikit-learn 1.9.1 / OpenCV 5.0.0 /
  scikit-image 0.26.0 / torch 2.14.0 (CPU).
* **Why it matters here:** feature extraction was shown to reproduce
  `feature_quality_dataset.csv` **bit-for-bit across different library
  versions** — that is the reproducibility claim the whole evaluation rests on.

---

## 3. Phase 1 — data + integrity (shared base)

```bash
python scripts/download_uieb.py     # ~1.6 GB tarball from the JJsnowx/UIEB_Dataset mirror
python scripts/validate_dataset.py  # image integrity, duplicates, pairing
```

| item | detail |
|---|---|
| writes | `dataset/raw-890/`, `dataset/reference-890/`, `dataset/challenging-60/` (~1.6 GB, **gitignored**) |
| verifies | 890/890 raw & reference, identical contiguous filenames `UIEB_0..UIEB_889` (`scripts/download_uieb.py:91-116`) |
| validate writes | `results/feature/dataset_validation_report.csv` (890 rows × 13 cols), `duplicate_groups.csv`, `phash_pairing_control.csv`, `plots/pairing_contact_sheet_A|B.png` |
| CHECKS 1–2 | shapes/channels/dtypes, raw↔reference dimension match, raw≠reference bytes, perceptual-hash pairing sanity (`scripts/validate_dataset.py:160-236`) |
| measured control | paired phash median **2.0** vs cross-pair **32.0** → the pairing is genuine, not assumed |
| gate | all critical checks PASS; 7 byte-identical duplicate groups enumerated (`UIEB_111/588, 479/488, 517/590, 627/770, 645/735, 653/783, 665/785`) |

**Re-run rule:** `dataset/` is gitignored, so a fresh clone must repeat Phase 1
before anything else can run.

---

## 4. Phase 2 — the split (shared base, frozen forever)

```bash
python scripts/make_split.py        # -> results/feature/data_split.csv
```

* Groups images by **SHA-256 of the raw file bytes**, so the 7 duplicate pairs
  can never straddle splits (`scripts/make_split.py:36-42`).
* Greedy fill in a seeded permutation (`RANDOM_STATE = 42`, 70/15/15).
* **CHECK 11 — asserted, not hoped:** the three id sets are pairwise disjoint and
  cover all 890 (`scripts/make_split.py:76-80`).
* Every later stage joins on `image_name` to this one file. There is no second
  split anywhere in the repository.

---

## 5. Phase 3a — classical enhancement (shared base)

```bash
python scripts/run_preprocessing.py     # -> dataset/preprocessed/ + preprocessing_log.csv
```

`src/preprocess.py::enhance` (`src/preprocess.py:98-110`), strictly in this order:

```
raw BGR → resize_to_width(600, INTER_AREA) → gray_world_white_balance
        → CLAHE(clip 2.5, 8x8) on the L channel of CIELAB
        → bilateralFilter(d=9, σcolor=75, σspace=75)
        → adaptive_gamma  γ = clip(mean_gray/128, 0.5, 2.0)  → preprocessed BGR uint8
```

* Every step is **per-image** — no cross-image statistic is fitted, which is why
  Phase 3 may legitimately touch all 890 images without leaking.
* The reference gets the *identical* geometric resize
  (`resize_reference_like_preprocessed`), which is what makes pixel-aligned
  SSIM/PSNR possible.
* gate: 890 outputs, all widths = 600, heights 266–901, γ ∈ [0.500, 1.412],
  mean γ 0.865 (`preprocessing_log.csv`; md5 `b104dd20…`).

---

## 6. Phase 4 — targets + features (shared base)

```bash
python scripts/build_feature_dataset.py
```

Per image identity:

```
preprocessed.png ─ imread(BGR) ─ cvtColor→RGB ─ src.features.extract_features ─► 25 floats
                                                  (statistical 6 | colour 7 | GLCM 8 | edge 4)
reference.png ─ resize_reference_like_preprocessed ─ compute_ssim_psnr(pre, ref) ─► ssim, psnr
row = image_name + 25 features + ssim + psnr      (27 columns, 890 rows)
```

* writes: `feature_dataset.csv` (26 cols) **and**
  `results/feature/feature_quality_dataset.csv` (27 cols) — the latter is the
  file every downstream stage reads.
* CHECKS 3–10 (`scripts/build_feature_dataset.py:96-140`): 890 rows == raw count,
  exactly 25 feature columns, zero missing values, zero duplicate ids, feature
  ranges printed, SSIM ∈ ~[0,1], PSNR finite, zero NaN/inf cells.
* **Reproducibility gate** (the one that matters): this CSV must come back
  **bit-identical** (md5 `e90a073f…`) after any environment reset. If it does,
  Phases 4–5 remain valid and need not be re-run. If it does not, every
  downstream number changes.

---

## 7. TRACK A — assessment, stage by stage

### A1. Ranking — `scripts/feature_ranking.py`

| | |
|---|---|
| reads | `feature_quality_dataset.csv` + `data_split.csv` |
| fits on | **train (623)** only — `RandomForestRegressor(200 trees, min_samples_leaf=2, seed 42)` |
| measures on | **val (134)** — `permutation_importance(n_repeats=5, scoring=neg_mean_squared_error)` |
| combined | per-target min-max normalisation → mean of the two → `importance_combined` |
| writes | `ranking_ssim.csv`, `ranking_psnr.csv`, `ranking_combined.csv` |
| asserts | (a) frame covers exactly the 25 canonical features; (b) **C1 guard** — the feature holding 1.000 in `ranking_combined.csv` must equal rank-1 in the matching per-target file, so three stale files can never coexist silently (`scripts/feature_ranking.py:137-165`) |
| gate | CHECK 13 — test ids present but never used for selection |

Result: `red_ratio` is rank 1 for **both** targets with normalised importance
1.000 (3.5× the runner-up `mean_red`, 0.288) — the physics of underwater red
attenuation, rediscovered by the selection stage. Val R² at this point: SSIM
0.5392, PSNR 0.3674.

### A2. Redundancy filter — `scripts/feature_correlation.py`

* Pearson 25×25 matrix on **train rows only**.
* List pairs with `|r| ≥ 0.90`; walk the combined ranking best→worst and drop a
  feature iff it correlates ≥0.90 with an **already-kept higher-ranked** one.
* Target correlation is never a removal criterion.
* writes: `feature_correlation_matrix.csv`, `highly_correlated_features.csv`,
  `selected_features.csv` (**14 survivors**), `removed_redundant_features.csv`.
* **11 features removed**, e.g. `ASM ← glcm_variance` (r = 1.0000), `energy ←
  glcm_variance` (0.936), `mean_green ← mean_blue` (0.983), `std ← entropy`
  (0.915), `rms_contrast ← entropy` (0.915), `colorfulness ← mean_saturation`
  (0.967), `gradient/edge_density ← keypoint_density` (0.929/0.937).

### A3. Subset sweep — `scripts/feature_subset_evaluation.py`

* Grid `SUBSET_SIZES = (15,12,10,8,6)` **clipped** to the survivor count →
  evaluated grid `{14,12,10,8,6}` (`scripts/feature_subset_evaluation.py:48-49`).
* One RF per target per k, fit on train, scored on **val**.
* writes `feature_subset_evaluation.csv`, `final_selected_features.csv`.

| k | SSIM R² | PSNR R² | avg R² (project-defined) |
|---|---|---|---|
| **14** | **0.5392** | **0.3674** | **0.4533** ← selected |
| 12 | 0.5220 | 0.3576 | 0.4398 |
| 10 | 0.5175 | 0.3585 | 0.4380 |
| 8 | 0.5222 | 0.3577 | 0.4400 |
| 6 | 0.5052 | 0.3320 | 0.4186 |

The curve is **flat** — k is not "optimal", the value of this stage is that
25 descriptors collapse to 14 (11 are redundant, 4 of them algebraically).

### A4. Models — `scripts/run_baselines.py`, `scripts/run_ablation.py`

Ordering matters: `run_ablation.py` **reads** `baseline_metrics.csv`, so
`run_baselines.py` must run first.

```bash
python scripts/run_baselines.py                  # RF + trains mlp / image_only / hybrid, evaluates test
python scripts/run_ablation.py                   # trains hybrid_all25 only, reuses A/B/D
python -m cnn.train --model hybrid --features final     # (what run_baselines calls internally)
python -m cnn.evaluate --run hybrid_final
```

Everything neural shares one protocol (`cnn/train.py`): Adam(1e−3, wd 1e−4),
**MSE on standardised targets**, early stopping patience 12 on val loss, max 80
epochs, **best** checkpoint kept, seeds 42/43/44 with `_s<seed>` suffixes
(`cnn/train.py:99-171`).

Fitting discipline, all on **train only** (`cnn/dataset.py:135-141`):
feature `StandardScaler`, target `StandardScaler`. Augmentation = Klein
four-group flips, **train split only**, chosen because the four transforms were
*measured* to leave all 25 features invariant to 2.2e−16 and to leave SSIM/PSNR
invariant (`cnn/dataset.py:59-82`). Rot90 and photometric jitter are excluded with
reasons stated in the code.

### A5. Test — read once

`cnn/evaluate.py` loads the best checkpoint + its train-fitted scalers, predicts
the sealed test split, inverse-transforms to original units, and writes
`results/cnn/<tag>/test_predictions.csv` + `metrics.json` (including a
4000-resample bootstrap 95% CI, the seed, whether augmentation was on, the best
epoch, and whether early stopping fired).

### A6. Independent verification — `scripts/verify_results.py`

```bash
python scripts/verify_results.py --run hybrid_final --expect-features 14
python scripts/verify_results.py --rf
```

31 `check()` call sites that re-derive every number with plain numpy (no
sklearn), re-assert the split (133 test ids exactly, disjoint, duplicate groups
whole), confirm the labels round-tripped unchanged, and — the clever part —
**probe the model's input usage**: perturbing the feature vector by +100 must
leave an image-only model bit-identical and must move a hybrid/MLP, and swapping
the image must move an image-only/hybrid model but not the MLP. Recorded as
35/35 PASS for the CNNs and 31/31 for the MLP.

### A7. Plots — `scripts/make_plots.py`

`plots/training_curve_<run>.png` (every run), `predicted_vs_actual_{ssim,psnr}.png`
(all models + RF, y=x reference), `model_comparison.png`, `ablation_comparison.png`.

### Track A results (sealed test, n = 133, all verified)

| run | model | inputs | SSIM R² | PSNR R² | avg R² | best epoch |
|---|---|---|---|---|---|---|
| `baseline_rf` | RandomForest | 14 features | 0.3576 | 0.2249 | 0.2913 | — |
| `mlp_final` | MLP | 14 features | 0.3255 | 0.1407 | 0.2331 | 22 |
| `image_only_nofeat` (A) | CNN | image | 0.4659 | 0.1996 | 0.3327 | 16 |
| `hybrid_all25` (C) | HybridCNN | image + 25 | 0.4003 | 0.1836 | 0.2919 | 9 |
| **`hybrid_final` (D)** | **HybridCNN** | **image + 14** | **0.4658** | **0.2211** | **0.3435** | **19** |

* D > A > C > B > MLP. **D beats C by +0.0516 avg R² with 11 fewer inputs** —
  direct evidence the selection stage does work (all 25 features actually *hurt*
  the CNN relative to pixels alone).
* D over A is **+0.0108**, entirely in PSNR; with ±~0.15 CIs at n=133 that gap is
  **not** statistically distinguishable. Never claim otherwise.
* Prediction range compression is real and reportable: RF predicts PSNR
  ∈ [12.14, 20.19] dB against a true [10.98, 28.82]; the hybrid never predicts
  above **18.73 dB**.

---

## 8. TRACK B — restoration / enhancement (Phase 3b)

The objective requires an *enhancement* component; every other model here is a
regressor that outputs two numbers and is structurally incapable of emitting an
image (`AdaptiveAvgPool2d(1)` collapses the spatial dims — stated in
`cnn/unet.py`'s docstring).

### B1. Train the U-Net — `python -m cnn.train_enhance`

| | |
|---|---|
| architecture | `EnhancementUNet(base=32)`: 3 levels (32→64→128), DoubleConv blocks, bilinear upsample + skip concat, 1×1 head, **Sigmoid** |
| params | **472,259**, asserted against `ENH_EXPECTED_PARAMS` at build time (`cnn/train_enhance.py:105-107`) |
| input/target | `dataset/preprocessed/<id>` → `dataset/reference-890/<id>` through the frozen aligned resize (`cnn/dataset_pairs.py:76-88`) |
| train | 623 pairs, 128×128 random crops, batch 8, **78 iterations/epoch**, L1 loss, Adam 2e−4, paired geometric augmentation (flips × rot90 = 8, identical on both images) |
| val | all 134 at **full resolution** every 5 epochs — mean SSIM/PSNR/L1 via `src/iqa.compute_ssim_psnr` |
| selection | early stopping on **val SSIM**, patience 4 validations (20 epochs), best checkpoint kept |
| leakage guards | train/val/test id sets asserted disjoint; train & val id **fingerprints** hashed into the checkpoint so an auditor can prove which images training saw (`cnn/train_enhance.py:93-115, 220-223`) |
| writes | `models/best_unet_128.pt` (**gitignored**), `results/enhancement/unet_128/train_history.csv` |
| actual run | **75 of 100 epochs**, best epoch **55** (val SSIM 0.809535, val PSNR 19.164), stopped at 75 after 4 stale validations; 99.5 s/epoch mean (~124 min training) |

### B2. Generate the enhanced images — `python -m cnn.enhance --split test`

* Writes `dataset/enhanced-test/<id>.png` for the 133 test ids (shape asserted
  equal to the input, dtype uint8 asserted) + `results/enhancement/unet_128/enhanced_test_manifest.csv`.
* Full-resolution inference pads to a multiple of 4 (the U-Net pools twice) and
  crops back: `pad_to_multiple` → `model` → `to_uint8` (`cnn/dataset_pairs.py:110-140`).
* ⚠ `ENHANCED_DIR` is **fixed** regardless of `--split`; use `--out` for smoke
  tests, see §12 risk D7.

### B3. Evaluate — `python scripts/evaluate_enhancement.py`

Scores three systems against the *same* aligned reference with the *same* metric
on the *same* 133 images; reads everything back **from disk** so a stale
checkpoint cannot produce a good-looking number. It also recomputes the classical
baseline and demands equality with the committed labels.

```
metric_definition_matches_committed_labels: true
max deviation vs committed:  SSIM 1.1e-16   PSNR 3.6e-15
```

| system | mean SSIM | median SSIM | mean PSNR |
|---|---|---|---|
| raw (resized only) | 0.7603 | 0.7814 | 17.063 dB |
| classical pipeline | 0.7636 | 0.7880 | 17.089 dB |
| **U-Net (`unet_128`)** | **0.8003** | **0.8163** | **19.324 dB** |

Paired test, U-Net vs classical (`results/enhancement/unet_128/metrics.json`):

| metric | delta | win / loss / tie | Wilcoxon p | paired bootstrap 95% CI |
|---|---|---|---|---|
| SSIM | **+0.03665** | 102 / 31 / 0 | < 1e-6 | [+0.0285, +0.0455] — excludes 0 |
| PSNR | **+2.2353 dB** | 121 / 12 / 0 | < 1e-6 | [+1.878, +2.588] — excludes 0 |

### B4. Independent verification — `python scripts/verify_enhancement.py`

53 checks, all re-derived from artefacts on disk: 133 PNGs present and *exactly*
the sealed test ids, shapes/dtypes, checkpoint id-fingerprints vs the frozen
split, enhanced files newer than the checkpoint, checkpoint config == frozen
Config A, `best_epoch == argmax(val_ssim)`, every metric reproduced, from-scratch
SSIM/PSNR agreement (20 images), the classical baseline equal to the committed
labels, and two anti-cheat checks — the images are **not** identical to the input
(it is not an identity map) and **not** byte-identical to the reference (no
target leak). Recorded as **53/53 PASS**.

### B5. Figures

`python scripts/make_enhancement_charts.py` → `plots/enhancement_training_curve.png`,
`plots/enhancement_per_image_delta.png` (CSV-only: works even with no images).
`python scripts/make_enhancement_samples.py` → raw | classical | U-Net | reference
strips (**needs the images back**).

### What Track B shows, stated honestly

* The learned enhancer is measurably better than the classical pipeline on the
  sealed test set (+0.037 SSIM, +2.24 dB PSNR, both CIs excluding zero) — this is
  the strongest statistical result in the project.
* It also beats the population whose quality Track A predicts: the classical
  labels are SSIM 0.7710 / PSNR 17.02 across all 890, while the U-Net reaches
  0.8003 / 19.324 on test.
* But: the classical pipeline barely beats *doing nothing but resizing*
  (0.7636 vs 0.7603 SSIM, 17.089 vs 17.063 dB) — and that comparison has **no
  paired test recorded** (see §12 risk D6). By these metrics, essentially all of
  the measured improvement comes from the learned model, not from the hand-designed
  chain.

---

## 9. Function-level data flow (one image, end to end)

**Track A — training/evaluation of the hybrid:**

```
imread(BGR) → src.preprocess.enhance
   resize_to_width → gray_world_white_balance → apply_clahe → apply_bilateral → adaptive_gamma
   → dataset/preprocessed/<id>.png
UIEBQualityDataset.__getitem__            cnn/dataset.py:114-133
   imread → cvtColor(BGR2RGB) → letterbox(224, INTER_AREA + black pad)
   → [augment_flips if train] → /255 → CHW float32
   → feat_scaler.transform(14 features)      (fit on train only)
   → target_scaler.transform([ssim, psnr])   (fit on train only)
HybridCNN.forward                         cnn/model.py:88-100
   ImageBranch: 4×ConvBlock(3→32→64→128→256) → GlobalAvgPool → 256-d
   FeatBranch : Linear(14→32) → ReLU → Dropout → 32-d
   concat 288 → Linear(288→128) → ReLU → Dropout(0.3) → Linear(128→2)
MSELoss(normalised) → early stop on val loss → best checkpoint
evaluate: no augmentation → inverse_transform → regression_metrics + bootstrap_r2_ci
```

**Track B — enhancement:**

```
load_pair(name)                            cnn/dataset_pairs.py:76-88
   preprocessed ↔ reference resized with the SAME frozen transform; shapes asserted equal
PairedEnhancementDataset.__getitem__       cnn/dataset_pairs.py:209-223
   paired_geometric(8 transforms, applied identically) → random 128×128 crop → to_tensor
EnhancementUNet.forward                    cnn/unet.py:82-87
   d1(H) → pool → d2(H/2) → pool → d3(H/4) → up+bottleneck+skip → u2(H/2) → up+skip → u1(H) → 1×1 → Sigmoid
L1Loss → full-res val every 5 epochs → best val-SSIM checkpoint
inference: pad_to_multiple(4) → model → crop back to (H,W) → to_uint8 → PNG
metrics : src.iqa.compute_ssim_psnr(enh, aligned_ref) → paired bootstrap + Wilcoxon
```

---

## 10. The leakage firewall — where every fit happens

| stage | train | val | test | reference images |
|---|---|---|---|---|
| 1 download/validate | ✓ | ✓ | ✓ | ✓ (dimensions / pairing only) |
| 2 split | ✓ | ✓ | ✓ | ✗ (ids only) |
| 3a preprocessing | ✓ | ✓ | ✓ | ✗ (per-image only) |
| 4 targets + features | ✓ | ✓ | ✓ | ✓ **only to compute labels** |
| A1 ranking | **fits** | permutes | ✗ | ✗ |
| A2 correlation | **fits** | ✗ | ✗ | ✗ |
| A3 k-sweep | **fits** | **chooses k** | ✗ | ✗ |
| A4/B1 modelling | **fits** | early stopping | ✗ | ✗ (Track B uses it as the *target*) |
| A5/B3 evaluation | ✗ | ✗ | ✓ **once** (RF, MLP, image-only, hybrid, hybrid_all25, U-Net, raw, classical) | ✓ to compute ground truth |

Three rules the code actually enforces (not just documents):
1. The reference is a **target source**, never a feature: scalers, augmentation
   and every model signature see only preprocessed images + features.
2. **Test is read once**, after all choices are frozen; k was chosen on val.
3. **Everything fitted is fitted on train** — two StandardScalers
   (`cnn/dataset.py:139-140`), the RFs, the CNN weights, the U-Net weights.

---

## 11. Artefact → claim map

| claim | artefact that proves it |
|---|---|
| 890 paired images, pairing genuine | `dataset_validation_report.csv`, `phash_pairing_control.csv`, `plots/pairing_contact_sheet_*.png` |
| 7 duplicate groups kept whole | `duplicate_groups.csv` |
| split is 623/134/133 and disjoint | `data_split.csv` (+ assert in `make_split.py`, re-asserted in `verify_results.py`) |
| preprocessing reproducible | `preprocessing_log.csv` md5 `b104dd20…` |
| 25 features × 890 rows, bit-reproducible | `feature_quality_dataset.csv` md5 `e90a073f…` |
| selection was train/val only, labels consistent | `ranking_*.csv` + the C1 assertion; `highly_correlated_features.csv`; `removed_redundant_features.csv` |
| final feature set | `final_selected_features.csv` (14, `red_ratio` first) |
| models were trained honestly | `results/cnn/<tag>/train_history.csv` (best epoch vs val loss) |
| test numbers, with CIs | `baseline_metrics.csv`, `ablation_results.csv`, `results/cnn/<tag>/metrics.json` |
| ablations A/B/C/D | `ablation_results.csv` |
| U-Net result + paired tests | `results/enhancement/unet_128/{metrics.json,test_per_image.csv}` |
| U-Net used the frozen config, test unseen | checkpoint config + id fingerprints, checked by `verify_enhancement.py` |
| figures | `plots/` (14 PNGs, including `enhancement_samples.png` added after this audit was written) |

---

## 12. Discrepancies found while doing this analysis

Ranked by how much damage they can do in a viva or a fresh clone.
**D1–D2 are real defects; D3–D5 are stale/missing documentation; D6–D8 are gaps
and footguns.**

### D1 — *"the enhanced PNGs are tracked"* is false, and they are gone
`.gitignore` claims *"U-Net enhanced outputs … TRACKED since 2026-09-20 … they
earn permanent storage in git"* and the HEAD commit message is *"Track
dataset/enhanced-test/: durability for the 133 enhanced PNGs"*. Evidence against:

```
$ git ls-tree -r HEAD --name-only | grep -c enhanced-test     → 0
$ ls dataset/enhanced-test | wc -l                            → 0
$ git check-ignore -v .../enhanced-test/UIEB_0.png            → (no match, so not ignored)
```

The pattern was correctly un-ignored but the files were **never added**, and they
are not in this working tree. Since `models/` is gitignored too (no `best_unet_128.pt`),
the entire Track B verification chain is **not reproducible from the repository**:
`scripts/evaluate_enhancement.py`, `scripts/verify_enhancement.py` and
`scripts/make_enhancement_samples.py` all fail on missing inputs; only the
CSV-based chart script runs. Recovery costs a ~124 min retrain + inference.
**Fix:** `git add -f UIE-fyp/dataset/enhanced-test` (~40 MB) — it is the only
artefact in this project that is expensive to regenerate.
(The same is true of every checkpoint: RF/MLP are cheap, each CNN is 25–40 min,
the U-Net 124 min. Decide deliberately what must be durable.)

### D2 — the frozen-config benchmark document does not exist
`src/config.py:126` cites `docs/enhancement-benchmark.md` as the measured
one-epoch benchmark (61.9 s/epoch, 1290 MB peak RSS, 78 iterations) from which
Config A was frozen. That file is **absent from both branches**. Worse, the
committed run's own `train_history.csv` shows **99.5 s/epoch mean (92–105 s)**,
i.e. ~1.6× the cited benchmark — so the number that justified the frozen config
cannot be reconciled with the run that used it. Write the file from
`ENH_*` + the history CSV, and state the discrepancy honestly.

### D3 — completion statuses are stale in three documents
| document | says | reality |
|---|---|---|
| `README.md` §18 (C2) | "CNN never run … **in progress**" | all four runs + ablation committed and independently verified |
| `docs/methodology-audit.md:472` | "Run the CNN suite (C2) — **IN PROGRESS**" | same |
| `docs/project-flow.md` §4 row 7, §5 Step 3 | "Plots **PENDING**" / "`make_plots.py` **STILL PENDING**" | 13 plots are committed, including `model_comparison.png` and `ablation_comparison.png` |
| `docs/project-flow.md` §5 Step 5 | "Fill README §12–§14 — **PENDING**" | still true: §12 has only the RF row; §13/§14 list the models but **no numbers** |

### D4 — the enhancement track is undocumented where it matters most
`README.md` §17 ("Reproducing everything") never mentions `cnn.train_enhance`,
`cnn.enhance`, `scripts/evaluate_enhancement.py`, `scripts/verify_enhancement.py`
or `scripts/verify_results.py`. `docs/architecture.md` has no entry for
`cnn/unet.py`, `cnn/dataset_pairs.py`, `cnn/train_enhance.py`, `cnn/enhance.py`
or the verification scripts. So the two-track structure of §0 — the single most
important thing about this repository — is visible only in `project-flow.md` and
in module docstrings.

### D5 — provenance: cited commits and branch no longer exist here
`project-flow.md` cites `f9a7832`, `97fac81`, `15e3d6c`, `f0d2f48` and branch
`arena/01a0a56a-uie-fyp`. `15e3d6c` and `f0d2f48` exist (reachable from
`origin/arena/01a0a56a-uie-fyp`, whose tip is `8029bfa` *"Recover the four
quality-prediction commits lost to a sandbox reset"*); `f9a7832` and `97fac81`
are gone. This checkout (`432ac4e`) is a **single squashed root commit with no
parent**, so there is no local history at all, and `.gitignore` differs between
the two branches. Any statement of the form "commit X contains Y" must be
re-verified against what is actually reachable.

### D6 — the one untested link in the enhancement claim
`evaluate_enhancement.py` computes raw metrics but runs the paired test only for
U-Net vs classical. **Classical vs raw** — i.e. "does the hand-designed chain do
anything at all?" — is computed and never tested. The gap it leaves is real:

| | mean SSIM | mean PSNR |
|---|---|---|
| raw (resize only) | 0.7603 | 17.063 |
| classical | 0.7636 | 17.089 |
| difference | +0.0034 | +0.026 dB |

Add one `bootstrap_mean_delta_ci` + `paired_wilcoxon` call (2 minutes) and report
it either way; it strengthens the U-Net claim and it is exactly the kind of
self-check this project otherwise does well.

### D7 — footguns in the ordering / interfaces
* `cnn/enhance.py` writes to a **fixed** `ENHANCED_DIR` for any `--split`,
  including `--split val` and `--limit 5`. Running either into the default
  directory silently mixes non-test images into `dataset/enhanced-test/`, which
  breaks `verify_enhancement.py` checks 1–2 ("exactly 133 PNGs, exactly the
  sealed ids"). Always pass `--out` for smoke tests.
* `run_ablation.py` **requires** `results/comparison/baseline_metrics.csv` —
  run `run_baselines.py` first, or re-run it with `--skip-train`.
* `baseline_metrics.csv` leaves the `*_r2_ci95_lo/hi` cells **empty** for the
  three neural rows; their CIs live in `results/cnn/<tag>/metrics.json` under
  `r2_ci95`. Do not read neural CIs from the CSV.
* `scripts/make_enhancement_charts.py` writes `Path("plots/...")` (CWD-relative)
  instead of `PLOTS_DIR` — it only works when run from the project root.
* `cnn/predict.py` only reconstructs `hybrid` and `image_only` checkpoints; an
  MLP run tag would raise.
* `scripts/feature_correlation.py` reads `ranking_combined.csv` with **no
  freshness guard**. If features are re-extracted and `feature_ranking.py` is not
  re-run first, the survivor set is derived from a stale ranking silently — the
  C1 assertion only catches inconsistency *within* one ranking run.
* Verification evidence (35/35, 53/53 PASS) exists only as console output quoted
  in prose. No `verification.txt` is committed, so "it was verified" is not
  independently checkable from the repository.

### D8 — the tracks are not connected (the honest limitation)
Track A predicts the quality of the **classical** pipeline only. It has never
been applied to Track B's outputs, so nothing in the repository demonstrates that
the quality estimator transfers to a different image producer (upgrade U6). The
label-distribution evidence says this is not a formality: the hybrid's predicted
PSNR never exceeds **18.73 dB** on its own test set, while the U-Net's mean PSNR
against the reference is **19.324 dB** — the enhancer now operates *above the
predictor's training-label range*, so scoring it with the frozen predictor would
be extrapolation, not evaluation.

---

## 13. Correct re-run order (fresh clone / after a sandbox reset)

`dataset/` and `models/` are gitignored; results, plots and docs are committed.
So the expensive path is only needed to *regenerate pixels or checkpoints*, never
to *read the numbers*.

**This checkout's state (verified):** `dataset/` contains only
`mirror_file_listing.json`; `dataset/enhanced-test/` is empty; `models/` does not
exist. Every committed CSV/JSON/PNG in `results/` and `plots/` is present.

```bash
# --- base (required before any model work) ---
python scripts/download_uieb.py            # ~1.6 GB
python scripts/run_preprocessing.py
python scripts/validate_dataset.py         # re-run AFTER preprocessing (fills preprocessed_* cols)
python scripts/build_feature_dataset.py    # GATE: must be bit-identical (md5 e90a073f…)
#   identical  -> Phases 2–5 stay valid, do NOT re-run them
#   different  -> re-run make_split → ranking → correlation → subset → models (everything changes)
python scripts/make_split.py               # GATE: CHECK 11

# --- Track A (only if features/selection changed, or to regenerate checkpoints) ---
python scripts/feature_ranking.py          # GATE: C1 assertion
python scripts/feature_correlation.py
python scripts/feature_subset_evaluation.py
python scripts/run_baselines.py            # TEST READ ONCE
python scripts/run_ablation.py
python -m cnn.evaluate --run <tag>         # if retraining per model
python scripts/verify_results.py --run <tag> --expect-features <0|14|25>
python scripts/make_plots.py

# --- Track B (needs ~124 min of CPU, then ~1 min of inference) ---
python -m cnn.train_enhance                # -> models/best_unet_128.pt
python -m cnn.enhance --split test         # -> dataset/enhanced-test/ (133 PNGs)
python scripts/evaluate_enhancement.py     # GATE: reproduces the committed classical labels
python scripts/verify_enhancement.py       # GATE: 53/53
python scripts/make_enhancement_charts.py
python scripts/make_enhancement_samples.py # needs the PNGs
```

---

## 14. Where the flow is incomplete (what to do next, in dependency order)

| order | item | why it is next | cost |
|---|---|---|---|
| 1 | **Make the artefacts durable** (D1) | one commit; it is the only thing that cannot be cheaply recovered | 5 min |
| 2 | **Write `docs/enhancement-benchmark.md`** (D2) and record the 61.9 vs 99.5 s/epoch discrepancy | the frozen-config claim currently has no evidence | 20 min |
| 3 | **Sync the statuses and fill README §12–§14** (D3, D4) | examiners read README first; it currently contradicts the repository | 1 h |
| 4 | **Add the classical-vs-raw paired test** (D6) | closes the last untested link in the enhancement claim | 2 min |
| 5 | **U6 — transfer test**: score a second producer's outputs (U-Net, or a cheap second classical chain) with the frozen predictor and check rank correlation with true SSIM; or retrain on U-Net outputs | turns "only validated on one pipeline" from a limitation into a result; note D8's range argument | 2–3 h |
| 6 | **U2 — 3 seeds (`CNN_SEEDS=(42,43,44)`)** on the neural models | single-seed differences at n=133 are noise; the plumbing already exists | 3–6 h unattended |
| 7 | **U4 — BRISQUE/NIQE baselines** on the same test split | answers "why not an off-the-shelf blind metric?" | 2–3 h |
| 8 | **U5 — bootstrap the ranking 200×** and report each feature's selection frequency | converts "red_ratio is rank 1" into a robustness claim | 1–2 h |
| 9 | Decide H4 (GLCM 1 angle vs 4) and U10 (Spearman sensitivity) and then **stop deciding** | both change features → full re-run of Phases 4–7; do them together or not at all | 1 h + re-extract |

---

## 15. Related documents

| file | role |
|---|---|
| `README.md` | the authoritative report (method validity §0, results §12–§14, architecture §15–§16, reproduction §17, defects §18, limitations §19) — **§12–§14 are still unfilled for the CNN models** |
| `docs/project-flow.md` | the roadmap: 9 phases, gates, the leakage firewall, ranked upgrades, thesis mapping, viva pack |
| `docs/architecture.md` | per-file explanation of `src/`, `cnn/`, `scripts/` (**pre-enhancement: no U-Net entries**) |
| `docs/plain-language-explanation.md` | the project with no assumed background |
| `docs/methodology-audit.md` | the independent audit: C1–C3, H1–H5, M1–M5, L1–L8 |
| **this file** | the as-built flow: what runs, in what order, with what evidence — and every place the docs disagree with the code |
