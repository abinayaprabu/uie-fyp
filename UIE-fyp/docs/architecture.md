# Architecture and concepts — file by file

What each file does, which technique it implements, and why that technique was chosen.
Read alongside `docs/methodology-audit.md`, which records what is currently broken.

## 1. System-level view

Three packages with strictly separated responsibilities:

```
src/      shared library — configuration, classical enhancement, feature
          extraction, metrics. No I/O orchestration, no ML training.
cnn/      PyTorch layer — dataset/loading + leakage control, model
          architectures, training protocol, evaluation, single-image inference.
scripts/  pipeline stages — one executable per stage, run in order. Each reads
          the previous stage's CSV artifacts and writes its own.
```

The pipeline is **file-mediated, not in-memory**: every stage persists a CSV, so any
stage can be re-run or audited independently, and the test split can be provably
untouched until the final stage.

Execution order:

```
download_uieb.py      -> dataset/{raw-890,reference-890,challenging-60}
validate_dataset.py   -> dataset_validation_report.csv, duplicate_groups.csv, contact sheets
run_preprocessing.py  -> dataset/preprocessed/, preprocessing_log.csv
build_feature_dataset.py -> feature_dataset.csv, feature_quality_dataset.csv (+ ssim/psnr targets)
make_split.py         -> data_split.csv            (group-aware, seed 42)
feature_ranking.py    -> ranking_{ssim,psnr,combined}.csv      [TRAIN fit, VAL permute]
feature_correlation.py-> feature_correlation_matrix.csv, highly_correlated_features.csv,
                        selected_features.csv, removed_redundant_features.csv   [TRAIN]
feature_subset_evaluation.py -> feature_subset_evaluation.csv,
                        final_selected_features.csv                [TRAIN fit, VAL score]
run_baselines.py      -> baseline_metrics.csv, rf_baseline_test_predictions.csv  [TEST, once]
run_ablation.py       -> ablation_results.csv                                    [TEST, once]
make_plots.py         -> plots/*.png
```

The **data-flow concept** that governs everything: a raw image is transformed by a
deterministic classical pipeline; the *reference* image is never a model input, it only
generates the two regression targets. So the model learns

> given an enhanced image and its descriptors, predict how close that enhancement is to
> the human-preferred UIEB reference.

That is **no-reference prediction of full-reference metrics** (blind IQA), not image
restoration.

## 2. `src/` — shared library

### `src/config.py`
**Concept: single source of truth for reproducibility.** Every stage imports parameters
from here, so it is impossible for preprocessing, feature extraction and the CNN to
disagree about a value. Holds paths, `RANDOM_STATE = 42`, split fractions (0.70/0.15/0.15),
preprocessing constants, RF and permutation settings, `CORR_THRESHOLD = 0.90`,
`SUBSET_SIZES`, CNN hyperparameters, and `FEATURE_NAMES_25` — the canonical feature order
that every downstream frame is indexed by.

### `src/preprocess.py`
**Concept: classical (non-learned) underwater image enhancement**, five ordered stages:

1. **`resize_to_width`** — aspect-preserving resize to width 600 with `INTER_AREA`.
   Area interpolation averages the source pixels falling inside each destination pixel,
   which is the correct kernel for *downsampling*; bilinear/bicubic would alias and
   corrupt the sharpness features computed later.
2. **`gray_world_white_balance`** — Buchsbaum's gray-world assumption: the average
   reflectance of a natural scene is achromatic, so each channel is scaled by
   `mean(all channels) / mean(this channel)` to equalise channel means. Underwater, red
   is absorbed within a few metres, leaving a blue/green cast; equalising the means
   removes that cast. Gains come from the image alone — no dataset statistic is fitted,
   so this stage cannot leak information across the split.
3. **`apply_clahe`** — Contrast-Limited Adaptive Histogram Equalisation applied to the
   **L channel of CIELAB**, then merged back. CLAHE equalises histograms within local
   tiles (8×8) rather than globally, recovering contrast in murky regions without
   blowing out bright ones; the clip limit (2.5) caps how far any histogram bin may be
   amplified, which is what stops it amplifying noise. Operating on L rather than on
   R/G/B independently is deliberate: it brightens without shifting hue, whereas
   per-channel CLAHE would distort the colour balance the previous step just corrected.
4. **`apply_bilateral`** — edge-preserving smoothing (d=9, σ_colour=75, σ_space=75).
   A bilateral filter weights neighbours by *both* spatial distance and intensity
   similarity, so it suppresses suspended-particulate noise and backscatter while
   leaving edges intact. Keeping edges matters because three of the 25 descriptors
   (edge density, gradient, Laplacian variance) measure exactly those edges.
5. **`adaptive_gamma`** — per-image tone mapping with `γ = clip(mean_gray / 128, 0.5, 2.0)`,
   applied through a 256-entry lookup table. Dark images get γ < 1 (brightened), bright
   images γ > 1 (pulled back), mid-tones stay near 1. Continuous, deterministic and
   image-local; the γ actually used is logged so the behaviour is auditable.

`resize_reference_like_preprocessed` applies the *identical* geometric transform to the
reference image. UIEB raw/reference pairs share dimensions, so this single documented
resampling step is what makes SSIM/PSNR pixel-aligned.

### `src/features.py`
**Concept: a 25-dimensional handcrafted descriptor covering four complementary aspects
of image quality.** No single feature is claimed to *be* quality; together they span
intensity, colour, texture and structure.

- **Statistical (6)** — `mean`, `std`, `variance`, `entropy`, `dynamic_range`,
  `rms_contrast`. `entropy` is Shannon entropy of the gray-level histogram and measures
  distributional complexity; `dynamic_range` is max−min; `rms_contrast` is the RMS
  deviation of gray values from their mean.
- **Colour (7)** — `mean_red/green/blue`, `colorfulness`, `red_ratio`,
  `mean_saturation`, `mean_value`. `colorfulness` is the Hasler–Süsstrunk opponent-colour
  metric: form `rg = R−G` and `yb = ½(R+G)−B`, then `√(σ_rg² + σ_yb²)`. `red_ratio` is
  `mean R / (mean G + mean B)`, a direct indicator of the red attenuation that dominates
  underwater degradation. Saturation and value come from HSV.
- **Texture (8)** — GLCM/Haralick properties from a gray-level co-occurrence matrix
  (distance 1, angle 0, 256 levels, symmetric, normalised): `contrast`, `correlation`,
  `energy`, `homogeneity`, `ASM`, `dissimilarity`, `glcm_entropy`, `glcm_variance`.
  The GLCM counts how often gray level *j* occurs at a fixed offset from level *i*, so it
  describes spatial texture rather than the histogram alone.
- **Edge/sharpness (4)** — `edge_density` (fraction of Canny(100,200) edge pixels),
  `gradient` (mean Sobel magnitude), `laplacian_variance` (the classic focus/sharpness
  measure — the Laplacian responds to high-frequency intensity change, so its variance
  falls when an image is blurred), `keypoint_density` (SIFT keypoint count / area).

**Implementation contracts** (each one is a defect that was already fixed here):
input must be RGB uint8, so every caller converts from OpenCV's BGR; colour arithmetic
casts to `float32` first because uint8 subtraction underflows and addition overflows
(`10 − 200 → 66`, `200 + 100 → 44`); one lazily-created SIFT instance is shared across
all images; and a final guard raises on any non-finite value rather than silently
imputing.

### `src/metrics.py`
**Concept: honest metric naming.** `regression_metrics` returns r2 / rmse / mae /
pearson_r / n. `combined_r2` is the mean of SSIM-R² and PSNR-R² and is documented as
**project-defined**, never as an image-quality metric — every table that uses it must
label it `avg_R2_SSIM_PSNR (project-defined)`. "Accuracy" is never reported, because it
is meaningless for a continuous regression target.

## 3. `cnn/` — PyTorch layer

### `cnn/model.py`
Four architectures, all emitting two numbers.

- **`ConvBlock`** — `Conv3×3 → BatchNorm2d → ReLU → MaxPool2d(2)`. Batch normalisation
  stabilises optimisation on a small dataset; each pool halves spatial resolution.
- **`ImageBranch`** — four ConvBlocks (3→32→64→128→256) then `AdaptiveAvgPool2d(1)`,
  i.e. **global average pooling**, producing a 256-d visual vector. GAP replaces the
  usual Flatten + huge Dense layer: at 14×14×256 a flatten would feed 50 k inputs into
  the head and dominate the parameter count. GAP collapses each channel to its spatial
  mean, which is what keeps the model at ~0.43 M parameters — deliberately lightweight
  for 623 training images.
- **`FeatBranch`** — `Linear(n→32) → ReLU → Dropout(0.2)` → 32-d.
- **`RegressionHead`** — `Linear(in→128) → ReLU → Dropout(0.3) → Linear(128→2)`.
- **`HybridCNN`** (the proposed model) — concatenates 256 + 32 = 288 and feeds the head.
  This is **late / feature-level fusion**: each modality is encoded independently, and
  only the compact representations are joined. The two outputs are
  `[ssim_normalised, psnr_normalised]`.
- **`ImageOnlyCNN`** — image branch + head, features ignored. Ablation A / baseline.
- **`FeatMLP`** — three-layer MLP on features only, kept in the same framework so all
  neural models share optimiser, initialisation and early-stopping protocol. Named MLP,
  never "CNN".

**No pretrained weights, by design**: loading ImageNet priors would confound the
image-only vs hybrid comparison, because a gain could come from pretraining rather than
from the handcrafted features. The comparison stays attributable to the thing being
studied.

### `cnn/dataset.py`
**Concept: leakage control at the data-loading boundary.**

- **`letterbox`** — aspect-preserving resize plus centred black padding to 224×224.
  Chosen over stretch-resize so the CNN's input transform does not distort the geometry
  that the texture and edge features describe.
- **Feature scaler** — `StandardScaler` fit on **train rows only** and persisted with
  the checkpoint. Fitting on all 890 would leak val/test distribution into training.
- **Target scaler** — a *second* `StandardScaler`, also train-only, applied to
  `[ssim, psnr]`. This is what makes a single MSE loss fair: SSIM lives in [0,1] while
  PSNR lives in roughly [10,40] dB, so raw MSE would be dominated by PSNR by orders of
  magnitude. After standardisation both have unit variance at training time, so equal
  weighting is justified rather than accidental.
- Joins to `data_split.csv` on `image_name` and raises `FileNotFoundError` if any
  preprocessed image is missing on disk — the CHECK-14 guard.

### `cnn/train.py`
**Concept: one protocol for every neural model, so comparisons are fair.** Adam
(lr 1e-3, weight decay 1e-4), MSE on normalised targets, early stopping on validation
loss (patience 12, max 80 epochs), best checkpoint retained rather than last, seeded
generator for a deterministic shuffle order. Saves `best_<tag>.pt`, both scalers as
joblib, and `train_history.csv`. `resolve_features` switches between the final subset
and all 25, which is how ablation C vs D is produced from the same code path.

### `cnn/evaluate.py`
**Concept: the test split is read exactly once.** Loads checkpoint + scalers, predicts
test, then `inverse_transform`s the targets back to original units so every reported
number is interpretable (SSIM in [0,1], **PSNR in dB**). Writes `test_predictions.csv`
and `metrics.json`.

### `cnn/predict.py`
Single-image inference: extracts the checkpoint's own feature list from the same
preprocessed image, letterboxes it, predicts, inverse-transforms, prints SSIM/PSNR.
This is the no-reference use case — it needs no reference image at inference time.

## 4. `scripts/` — pipeline stages

### `scripts/download_uieb.py`
Acquires UIEB from the GitHub mirror `JJsnowx/UIEB_Dataset` (the official Google Drive
links are unreachable from some sandboxes) as one resumable `curl` tarball, extracts
into `raw-890 / reference-890 / challenging-60`, and verifies file counts plus raw↔reference
filename correspondence.

### `scripts/validate_dataset.py`
**Concept: pairing is verified, not assumed.** Two duplicate detectors — SHA-256 for
byte-identical files and a 64-bit DCT perceptual hash for near-identical re-saves. The
pHash is also used as a **pairing-plausibility test**: genuinely paired raw/reference
images must have a small Hamming distance, while a deliberately mismatched control
(each raw compared against a reference shifted by +13 positions) must be much larger.
If the paired median is not smaller than the control median, validation fails. Also
writes human-verifiable side-by-side contact sheets.

### `scripts/make_split.py`
**Concept: group-aware splitting.** Identities are grouped by the SHA-256 of their raw
file, groups are permuted with seed 42, then greedily filled train → val → test so that
duplicate members always land in the same split. Without this, two byte-identical images
in train and test would make the test score meaningless. Asserts the three splits are
pairwise disjoint and jointly exhaustive (CHECK 11).

### `scripts/build_feature_dataset.py`
Per paired identity: extract the 25 features from the preprocessed image, resize the
reference identically, compute SSIM (`channel_axis=2, data_range=255`) and PSNR. Runs
CHECKS 3–10 (row count, column count, no missing values, no duplicate ids, feature
ranges, SSIM within [0,1], finite PSNR, no NaN/inf). Uses **no split information at
all** — splitting is a strictly downstream concern.

### `scripts/feature_ranking.py`
**Concept: held-out permutation importance, not impurity importance.** A
`RandomForestRegressor(200, min_samples_leaf=2, seed 42)` is fit per target on **TRAIN**,
then `permutation_importance` runs on **VAL** (`n_repeats=5`, `scoring=neg_mean_squared_error`).
Permutation importance measures the rise in held-out error when a column is shuffled, so
it reflects actual predictive contribution — unlike MDI/impurity importance, which is
computed on training splits and is biased toward high-cardinality features. Per-target
importances are min-max normalised to [0,1] and averaged into a combined rank.

### `scripts/feature_correlation.py`
**Concept: unsupervised redundancy removal in rank order.** A 25×25 Pearson matrix is
computed on **TRAIN rows only**; all `|r| ≥ 0.90` pairs are recorded; then a greedy walk
proceeds from best combined rank to worst and drops a feature if it correlates ≥ 0.90
in absolute value with an already-kept, higher-ranked feature. Absolute value means
*negative* redundancy also counts. Crucially, correlation with the **target** is never a
removal criterion — the filter removes feature-to-feature duplication only, so it cannot
 smuggle label information into an unsupervised step.

### `scripts/feature_subset_evaluation.py`
Top-k of the survivors for each k in `SUBSET_SIZES`, clipped to the survivor count; one
RF per target fit on TRAIN and scored on VAL; the best k by project-defined average val
R² is written to `final_selected_features.csv`. Test metrics are deliberately **not**
computed here.

### `scripts/run_baselines.py`
The four-way comparison — `baseline_rf`, `mlp_final`, `image_only_nofeat`,
`hybrid_final` — consolidated into `baseline_metrics.csv`. The RF is trained in this
script; the neural models are trained via `cnn.train` and evaluated via `cnn.evaluate` so
that all four share one joint test evaluation. It prints the honest-reporting rule:
if the hybrid does not beat the baselines, that *is* the finding, and test must not be
re-tuned to force a win.

### `scripts/run_ablation.py`
The experiment that answers the project's central research question — *does feature
selection actually improve CNN-based quality prediction?*

| ablation | model | features |
|---|---|---|
| A | `image_only_nofeat` | none |
| B | `baseline_rf` | final selected |
| C | `hybrid_all25` | all 25 |
| D | `hybrid_final` | final selected |

A, B and D are reused from the baseline run (identical models, identical test split) so
only C is trained here — reuse without retraining keeps the comparison exact rather than
merely similar. **C vs D is the direct test of the feature-selection contribution.**

### `scripts/make_plots.py`
Matplotlib only (no seaborn). Training curves per run, predicted-vs-actual scatter per
target with an ideal diagonal overlay for every model, and grouped R²/RMSE bars from
`baseline_metrics.csv`. All three degrade gracefully when the CNN artifacts do not exist
yet.

## 5. The concepts that hold the project together

1. **Deterministic enhancement, learned assessment.** The classical pipeline has no
   fitted parameters, so it contributes no leakage; all learning happens in the quality
   predictor.
2. **References are targets, never inputs.** This is what makes the task no-reference
   prediction of full-reference metrics.
3. **Selection strictly inside train+val.** Ranking fits on train and permutes on val;
   correlation uses train; the k sweep scores on val; test is read once, at the end.
4. **Group-aware splitting** so byte-identical duplicates cannot straddle the boundary.
5. **Normalised multi-output regression** so two targets on incompatible scales share
   one loss without one silently dominating.
6. **Ablation against the selection claim.** The project does not assume feature
   selection helps; C vs D is designed to find out, and the code commits in advance to
   reporting a negative result honestly.
