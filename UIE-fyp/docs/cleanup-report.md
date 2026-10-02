# Repository cleanup report (2026-10-02)

*This is the FILE | STATUS | REASON table required before any cleanup. The rule
applied was: **archive, do not delete**. Only machine-generated cache files were
deleted. No dataset, no verified result, no useful documentation, and no code
required for reproducibility was removed.*

## 1. The table

| File / directory (old path) | Status | Reason |
|---|---|---|
| `cnn/model.py` | ARCHIVE → `archive/quality_prediction/code/cnn/model.py` | quality-prediction heads (FeatMLP / ImageOnlyCNN / HybridCNN); not part of the enhancement architecture |
| `cnn/train.py` | ARCHIVE | trainer for those predictors |
| `cnn/evaluate.py` | ARCHIVE | test evaluation for those predictors |
| `cnn/predict.py` | ARCHIVE | single-image prediction CLI for those predictors |
| `scripts/run_baselines.py` | ARCHIVE | trains the four predictor baselines (superseded experiment) |
| `scripts/run_ablation.py` | ARCHIVE | predictor-side ablation (image-only vs hybrid *predictors*) |
| `scripts/feature_ranking.py` | ARCHIVE | RF permutation ranking computed over all 890 rows — superseded by `stats/selector.py`, which is train/val only and leakage-audited |
| `scripts/feature_subset_evaluation.py` | ARCHIVE | feature-subset search for the predictors |
| `scripts/feature_correlation.py` | ARCHIVE | correlation study over all rows; superseded by `stats/correlation.py` (train only) |
| `scripts/make_plots.py` | ARCHIVE | plots for the prediction track; the final plots are produced separately |
| `scripts/verify_results.py` | ARCHIVE | verifies the predictor runs (it imports `cnn.model`); the final evaluation has its own checks (`leakage_audit.py`, `evaluate_hybrid.py`, `verify_enhancement.py`) |
| `results/cnn/**` (4 runs) | ARCHIVE | predictor metrics/histories; cited as *prior work* in the report |
| `results/comparison/**` | ARCHIVE | predictor baseline/ablation tables; cited as *prior work* |
| `plots/model_comparison.png`, `predicted_vs_actual_{ssim,psnr}.png`, `training_curve_{mlp_final,hybrid_all25,hybrid_final,image_only_nofeat}.png`, `ablation_comparison.png` | ARCHIVE | predictor-track figures |
| `__pycache__/`, `*.pyc` | DELETE | machine-generated caches (regenerated automatically) |

Everything not listed above was **KEEP**, in particular:

- `dataset/` — UIEB raw/reference/preprocessed/enhanced-test (never touched);
- `src/{config,preprocess,features,iqa,metrics,nr_metrics}.py` — validated
  preprocessing, 25-feature extraction, metrics;
- `stats/**` — the leakage-safe Stage-A selection;
- `results/{feature,statistics,feature_selection,metrics,enhancement/unet_128*}` —
  labels, split, selection evidence, sealed-test numbers, U-Net baseline;
- `cnn/{dataset,dataset_pairs,unet,train_enhance,enhance}.py` — data plumbing
  and the existing U-Net baseline;
- `cnn/feature_guided/**` — the proposed model;
- all documentation.

## 2. Structure of the final pipeline

The specification suggested `cnn/feature_guided/{encoder,feature_conditioning,
decoder,model,dataset,train,enhance}.py`. The implemented layout is exactly
that, with three names kept for consistency with the existing repository:

| Suggested | Actual |
|---|---|
| `cnn/feature_guided/encoder.py` | `cnn/feature_guided/encoder.py` |
| `cnn/feature_guided/feature_conditioning.py` | `cnn/feature_guided/conditioning.py` |
| `cnn/feature_guided/decoder.py` | `cnn/feature_guided/decoder.py` |
| `cnn/feature_guided/model.py` | `cnn/feature_guided/model.py` |
| `cnn/feature_guided/dataset.py` | `cnn/feature_guided/dataset.py` (+ `losses.py`) |
| `cnn/feature_guided/train.py` | `cnn/feature_guided/train.py` (`python -m cnn.feature_guided.train`) |
| `cnn/feature_guided/enhance.py` | `cnn/feature_guided/enhance.py` (`python -m cnn.feature_guided.enhance`) |

`src/` is **not** split into `preprocessing/ features/ statistics/ metrics/`
sub-packages: the existing flat modules (`src/preprocess.py`, `src/features.py`,
`src/iqa.py`, `src/metrics.py`, `src/nr_metrics.py`) plus the `stats/` package
already are the single source of truth, and re-organising them would either
duplicate code or break the verified scripts. The mapping is:

| Suggested | Actual |
|---|---|
| `src/preprocessing/` | `src/preprocess.py` (+ `scripts/run_preprocessing.py`) |
| `src/features/` | `src/features.py` (+ `scripts/build_feature_dataset.py`) |
| `src/statistics/` | `stats/` package (+ `scripts/run_statistics.py`) |
| `src/metrics/` | `src/iqa.py`, `src/metrics.py`, `src/nr_metrics.py` |

`results/` now contains `feature/`, `statistics/`, `feature_selection/`,
`metrics/`, `enhancement/`; `checkpoints/` is not used because trained weights
live next to their run (`results/enhancement/<run>/best_<run>.pt`, kept in git)
with the repo's existing `models/` convention as the working location.

## 3. What was deleted

Only `__pycache__/` directories and `.pyc` files inside the project (excluding
the virtualenv). Nothing else. The dataset, the U-Net baseline, its verified
results, the labels, the split, the 25-feature extraction and all documentation
are intact.
