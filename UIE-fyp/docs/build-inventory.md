# Build inventory — existing repo, reuse plan, and what must be built

*Written 2026-10-02 for the LOCKED specification. **Nothing is redesigned here.**
This is step 1–2 of the required build order: inspect the repository, report what
exists, what will be reused, what will be added, and what will be modified.*

Repository: `abinayaprabu/uie-fyp`, session branch `arena/01a0bd95-uie-fyp`
(tip `0d44951`). Project root inside the repo: `UIE-fyp/`.

---

## 1. Existing files and components found (verified, with line anchors)

### `src/` — shared library (frozen, do not rewrite)

| file | contents | status |
|---|---|---|
| `src/config.py` (153 lines) | single source of truth: paths, `RANDOM_STATE=42`, split fractions 70/15/15, preprocessing constants, RF/permutation settings, `CORR_THRESHOLD=0.90`, `FEATURE_NAMES_25`, CNN + U-Net hyperparameters | **frozen** |
| `src/preprocess.py` (125 lines) | `resize_to_width`, `gray_world_white_balance`, `apply_clahe`, `apply_bilateral`, `adaptive_gamma`, `enhance`, `resize_reference_like_preprocessed` — the exact 5-step pipeline with CLAHE 2.5/(8,8), bilateral 9/75/75, γ∈[0.5,2.0] | **frozen — matches spec §24 exactly** |
| `src/features.py` (142 lines) | `extract_features(rgb_uint8) → dict` of exactly the 25 descriptors (statistical 6, colour 7, GLCM 8, edge 4) | **frozen — matches spec §9 exactly** |
| `src/iqa.py` (95 lines) | `compute_ssim_psnr` (skimage, uint8, `data_range=255`), plus `psnr_from_scratch`, `ssim_from_scratch` as independent cross-checks | **frozen — the one SSIM/PSNR definition** |
| `src/metrics.py` (117 lines) | `regression_metrics`, `bootstrap_r2_ci`, `bootstrap_mean_ci`, `bootstrap_mean_delta_ci`, `paired_wilcoxon` | reused as-is for Stage C statistics |

### `cnn/` — PyTorch layer

| file | contents | relevance to the locked architecture |
|---|---|---|
| `cnn/model.py` (135 lines) | `ConvBlock` (Conv3×3→BN→ReLU→MaxPool2), `ImageBranch` (4 blocks 3→32→64→128→256 + GAP → 256-D), `FeatBranch` (Linear→ReLU→Dropout→32-D), `RegressionHead`, `HybridCNN`, `ImageOnlyCNN`, `FeatMLP`, `count_params` | **`ConvBlock`, `ImageBranch` and `FeatBranch` are exactly the locked spec's blocks, encoder and feature branch** — reused as components (the quality-prediction models themselves are prior work, not the proposed model) |
| `cnn/unet.py` (92 lines) | `DoubleConv`, `EnhancementUNet` (3-level, skips, bilinear up, 1×1 head, Sigmoid, 472,259 params asserted) | the decoder *pattern* (upsample → concat skip → double conv) is reused; the new decoder is 4-stage to match the locked encoder |
| `cnn/dataset_pairs.py` (223 lines) | `aligned_reference`, `load_pair`, `to_tensor`, `to_uint8`, `pad_to_multiple`, `enhance_image`, `paired_geometric` (8 paired transforms), `split_ids`, `ids_fingerprint`, `PairedEnhancementDataset` | **reused nearly verbatim** for the hybrid's paired dataset; `ids_fingerprint` gives the leakage audit trail |
| `cnn/dataset.py` (141 lines) | `letterbox` (224, INTER_AREA + black padding), `augment_flips` (Klein four-group), `UIEBQualityDataset`, `fit_scalers` (train-only StandardScaler) | **`letterbox` and `fit_scalers` are exactly what the locked hybrid needs** |
| `cnn/train.py`, `train_enhance.py`, `evaluate.py`, `enhance.py`, `predict.py` | the two existing training loops, evaluation and inference | patterns reused (seeding, best-checkpoint, scaler persistence, metric writing); not modified |

### `scripts/` — 16 pipeline stages (all working, committed)

`download_uieb` · `validate_dataset` · `run_preprocessing` · `build_feature_dataset`
· `make_split` · `feature_ranking` · `feature_correlation` · `feature_subset_evaluation`
· `run_baselines` · `run_ablation` · `make_plots` · `evaluate_enhancement`
· `verify_enhancement` · `verify_results` · `make_enhancement_charts` · `make_enhancement_samples`

### `results/` — committed evidence

| path | contents |
|---|---|
| `results/feature/feature_quality_dataset.csv` | **890 rows × 27 cols: `image_name` + 25 features + `ssim` + `psnr`** — the Stage-A statistics input |
| `results/feature/data_split.csv` | frozen group-aware split: 623 train / 134 val / 133 test, seed 42 |
| `results/feature/{ranking_*,selected_features,highly_correlated_features,removed_redundant_features,final_selected_features,feature_subset_evaluation}.csv` | the OLD (RF-permutation) selection chain, including the **OLD 14** |
| `results/feature/preprocessing_log.csv` | 890 rows: gamma, out_h, out_w — widths all 600, heights 266–901 |
| `results/feature/ground_truth_quality_scores.csv` | image_name, split, ssim_actual, psnr_actual (slide-ready) |
| `results/cnn/*`, `results/comparison/*` | the four trained+verified quality-prediction models and the A/B/C/D ablation (prior work) |
| `results/enhancement/unet_128/`, `unet_128_repro/` | the existing U-Net baseline: metrics.json, per-image CSV, training history, manifests |
| `dataset/enhanced-test/*.png` | 133 committed enhanced images from the existing U-Net (independently pixel-verified) |

### `docs/`

`exact-flow.md` (as-built flow + discrepancy audit) · `project-flow.md` (roadmap)
· `architecture.md` · `plain-language-explanation.md` · `methodology-audit.md`
· `stage2-3-plan.md` (the plan for this new direction).

### **Does any statistics or hybrid-enhancer code exist?**

**No.** A repo-wide search for `statistic|hybrid_enh|fusion|decoder|uciqe|uiqm`
returns nothing. Both new stages start from zero code.

---

## 2. What will be reused, unchanged

| locked-spec component | reused artifact | exact match |
|---|---|---|
| Preprocessing (§24) | `src/preprocess.py` (all five steps + parameters) | identical values |
| 25 features (§9) | `src/features.py::extract_features` | same 25 names, same order |
| Feature table (§9) | `results/feature/feature_quality_dataset.csv` | `image_name` + 25 + SSIM + PSNR |
| Split, seed 42 (§20) | `results/feature/data_split.csv` + `make_split.py` | 623/134/133, group-aware |
| SSIM/PSNR definition (§21) | `src/iqa.py` | one definition project-wide |
| CNN encoder (§4) | `cnn/model.py::ConvBlock`, `ImageBranch` | 3→32→64→128→256, GAP → 256-D |
| Feature branch (§5) | `cnn/model.py::FeatBranch` | Linear→ReLU→Dropout(0.2)→32-D |
| Letterbox to 224 (§25) | `cnn/dataset.py::letterbox` | aspect-preserving + black pad |
| Train-only scaler (§20) | `cnn/dataset.py::fit_scalers` | fitted on train rows only |
| Paired dataset + augmentation (§26) | `cnn/dataset_pairs.py` | paired geometric transforms, identical on input/target |
| Leakage audit | `ids_fingerprint`, `split_ids` | hashes stored in the checkpoint |
| Evaluation statistics (§15) | `src/metrics.py` | bootstrap CI + paired Wilcoxon |
| U-Net baseline (§27) | `results/enhancement/unet_128/` + its checkpoint recipe | reported separately as baseline |
| Training hyperparameters (§26) | `CNN_LR=1e-3`, `CNN_WEIGHT_DECAY=1e-4`, `CNN_BATCH_SIZE=16`, `CNN_MAX_EPOCHS=80`, `CNN_PATIENCE=12`, `RANDOM_STATE=42` | **already equal to the spec** |

---

## 3. What is new (to be built)

```
statistics/                     NEW package  (Stage A)
    descriptive.py              mean/median/sd/variance/min/max/IQR + Shapiro/QQ inputs
    hypothesis_tests.py         f_regression per feature x target (F, raw p)  [primary]
    multiple_testing.py         Benjamini-Hochberg FDR (+ Bonferroni sensitivity)
    correlation.py              Pearson r, Spearman rho (effect sizes), feature-feature matrix
    redundancy.py               |r| >= 0.90 feature-feature rule, keep higher-ranked
    selector.py                 rank aggregation -> NEW 14 (cap documented)

model/hybrid/                   NEW package  (Stage B, ONE model)
    encoder.py                  HybridEncoder      : 4 blocks -> skips + 256x14x14 + GAP 256-D
    feature_branch.py           HybridFeatBranch   : 14 -> 32-D  (wraps existing FeatBranch)
    fusion.py                   Fusion             : 288 -> 256, broadcast-condition the bottleneck
    decoder.py                  EnhancementDecoder : 14->28->56->112->224, skips, 3ch Sigmoid
    hybrid_enhancer.py          HybridEnhancer     : the single proposed model
    dataset.py                  HybridDataset      : (image 224 letterboxed, 14 features, target)

metrics/nr_metrics.py           NEW: UIQM and UCIQE from the published definitions (cited)

scripts/
    run_statistics.py           Stage A end-to-end  -> results/statistics/*
    make_stat_plots.py          the 10 required visualisations
    train_hybrid.py             Stage B training
    evaluate_hybrid.py          Stage C: PSNR/SSIM/UIQM/UCIQE + paired tests vs raw/classical/U-Net
    verify_hybrid.py            the independent-verification pattern used for the U-Net

docs/
    architecture_explanation.md, statistical_analysis_explanation.md,
    feature_selection_explanation.md, training_explanation.md,
    metrics_explanation.md, viva_questions.md

results/statistics/ ... and the existing results/ tree for stage B/C outputs
```

## 4. What existing files will be modified (append-only)

| file | change | risk |
|---|---|---|
| `src/config.py` | **append** a `HYBRID_*` block (input 224, 4 blocks, widths 32/64/128/256, feature 14, hidden 32, fused 288, fusion 256, decoder widths, loss weights) and `STATS_*` constants | none — no existing constant changes |
| `requirements.txt` | possibly add nothing (scipy + scikit-learn already provide Shapiro, f_regression, Pearson, Spearman, and FDR support) | none |
| `README.md` | later: a section pointing to the new stages | none |
| *everything else* | **untouched** — no existing script, module, result file or plot is edited or deleted | — |

**Explicitly NOT modified:** `src/preprocess.py`, `src/features.py`, `src/iqa.py`,
`src/metrics.py`, `cnn/model.py`, `cnn/dataset.py`, `cnn/dataset_pairs.py`,
`cnn/unet.py`, `results/feature/*`, `results/enhancement/*`, `dataset/enhanced-test/*`.

---

## 5. Locked-architecture confirmation (read-back)

I will build exactly this, and nothing else:

1. Stage A: 25 features → descriptive stats → regression F-test per (feature, target)
   → raw p → Benjamini-Hochberg adjusted p (α = 0.05) → Pearson/Spearman effect sizes
   → feature–feature Pearson |r| ≥ 0.90 redundancy → documented rank aggregation → 14.
   Train rows only. No t-test, no categorical ANOVA, no χ², no user-invented groups.
2. Stage B: **ONE** hybrid enhancement model —
   image branch (CNN encoder → GAP → 256-D) **and** the 256×14×14 spatial bottleneck;
   feature branch (14 → StandardScaler(train) → 32-D);
   concatenate 288-D → Linear(288→256) fusion → broadcast-condition the bottleneck;
   U-Net decoder with skips 14→28→56→112→224 → 3-channel enhanced image.
   L1 loss (λ_SSIM optional and configurable). No attention, no transformer, no GAN,
   no pretrained backbone, no extra branches, no extra heads.
3. Stage C: enhanced image → PSNR, SSIM (full-reference vs UIEB reference) and
   UIQM, UCIQE (no-reference) → compared against raw, classical preprocessing and
   the existing U-Net baseline, with paired tests.
4. Behaviour: I will not create additional models, will not turn enhancement into
   quality prediction, and will stop and ask before any architecture change.

---

## 6. Final architecture diagram (locked)

```
                    UIEB raw 890 + reference 890
                                │
                     PREPROCESSING (src/preprocess.py, frozen)
          resize 600 → gray-world → CLAHE(L) → bilateral → adaptive gamma
                                │
                 ┌──────────────┴───────────────┐
                 ▼                              ▼
      25 FEATURES (src/features.py)     SSIM, PSNR vs reference (src/iqa.py)
                 └──────────────┬───────────────┘
                                ▼
                    feature_quality_dataset.csv (890 × 27)
                                │
              STATISTICAL ANALYSIS (TRAIN rows only, n=623)
        descriptive → regression F-test → raw p → BH-FDR (α=0.05)
        → Pearson r / Spearman ρ (effect size) → |r|≥0.90 redundancy
        → documented rank aggregation
                                │
                         14 SELECTED FEATURES
                                │                    │
                                │                    │
                    underwater image                 │
                 (letterboxed 3×224×224)             │
                                ▼                    ▼
                     IMAGE BRANCH               FEATURE BRANCH
                  Conv 3→32  (32×112×112)      StandardScaler(train)
                  Conv 32→64 (64×56×56)          (fit on train only)
                  Conv 64→128 (128×28×28)              ▼
                  Conv 128→256 (256×14×14)      Linear 14→32 → ReLU
                                │                 → Dropout(0.2)
                    ┌───────────┴──────────┐           ▼
                    ▼                      ▼        32-D
              GAP → 256-D            spatial 256×14×14
                    │                      │           │
                    └───────────┬──────────┘           │
                                │                      │
                                └──────────┬───────────┘
                                           ▼
                               CONCATENATE  256 + 32 = 288-D
                                           ▼
                                 FUSION  Linear(288→256) → ReLU
                                           ▼
                        broadcast 256 → 256×14×14, combine with bottleneck
                                           ▼
                              ENHANCEMENT DECODER (skips from encoder)
                        14→28 (+128×28×28) → 56 (+64×56×56)
                             → 112 (+32×112×112) → 224
                                           ▼
                                Conv 1×1 → 3 channels → Sigmoid
                                           ▼
                              ENHANCED IMAGE 3×224×224
                                           ▼
                         un-letterbox → resize to (H, W)  [H 266–901, W 600]
                                           ▼
              ┌────────────────────────────┴───────────────────────────┐
              ▼                        ▼                ▼              ▼
            PSNR                     SSIM           UIQM           UCIQE
      (vs reference,            (vs reference,   (no-reference)  (no-reference)
       full-reference)           full-reference)
              │                        │                │              │
              └────────────────────────┴────────────────┴──────────────┘
                                           ▼
                    compare: RAW · CLASSICAL · U-NET BASELINE · HYBRID
                              (paired tests, TEST split once)
```

## 7. Exact tensor dimensions (locked architecture)

| stage | tensor | dims |
|---|---|---|
| input image (letterboxed) | RGB | 3 × 224 × 224 |
| encoder block 1 | Conv 3→32, BN, ReLU, MaxPool | 32 × 112 × 112 |
| encoder block 2 | Conv 32→64, BN, ReLU, MaxPool | 64 × 56 × 56 |
| encoder block 3 | Conv 64→128, BN, ReLU, MaxPool | 128 × 28 × 28 |
| encoder block 4 | Conv 128→256, BN, ReLU, MaxPool | **256 × 14 × 14** |
| image vector | Global Average Pooling | **256** |
| input features | 14 selected (train-fitted scaler) | **14** |
| feature branch | Linear(14→32) → ReLU → Dropout(0.2) | **32** |
| concatenation | 256 ⊕ 32 | **288** |
| fusion | Linear(288→256) → ReLU | **256** |
| conditioning | 256 broadcast → add to bottleneck | 256 × 14 × 14 |
| decoder up 1 | upsample → concat skip(128) → Conv-BN-ReLU | 128 × 28 × 28 |
| decoder up 2 | upsample → concat skip(64) → Conv-BN-ReLU | 64 × 56 × 56 |
| decoder up 3 | upsample → concat skip(32) → Conv-BN-ReLU | 32 × 112 × 112 |
| decoder up 4 | upsample → Conv-BN-ReLU | 16 × 224 × 224 |
| output head | Conv 1×1 → 3, Sigmoid | **3 × 224 × 224** |
| after un-letterbox | resize to preprocessed geometry | 3 × H × W (H 266–901, W 600) |
| metrics | vs aligned UIEB reference | PSNR, SSIM, UIQM, UCIQE |

## 8. Build order (following the spec's steps 1–14)

| # | step | state |
|---|---|---|
| 1–2 | inspect repo, report what exists | **this document** |
| 3 | run/verify preprocessing (needs 1.6 GB re-download: `dataset/raw-890`, `reference-890`, then `run_preprocessing.py`; gate = `producing feature_quality_dataset.csv` bit-identically) | pending |
| 4 | verify 25-feature extraction (gate: reproduces the committed CSV) | pending |
| 5–6 | statistics package → real results in `results/statistics/` | pending |
| 7 | **OLD 14 vs NEW 14 comparison → STOP AND ASK** (spec §19) | pending |
| 8 | hybrid model code (`model/hybrid/*`) | pending |
| 9 | shape/unit tests (print every tensor shape) | pending |
| 10 | tiny overfit test (8–16 images, loss must decrease) | pending |
| 11 | full training (compute plan in §9 below) | pending |
| 12 | evaluate PSNR/SSIM/UIQM/UCIQE | pending |
| 13 | visual comparisons (RAW / CLASSICAL / HYBRID / REFERENCE + the four metrics per image) | pending |
| 14 | explanation documents + viva answers (from the real results) | pending |

## 9. Environment reality (measured, not assumed)

* `dataset/`: only `enhanced-test/` (133 committed PNGs) + `mirror_file_listing.json`.
  **raw-890, reference-890, preprocessed, challenging-60 are empty; `models/` is empty;
  the Python venv is absent** — the sandbox was reset. The 1.6 GB UIEB mirror
  re-downloads in ~1–2 min at the measured ~24 MB/s, preprocessing of 890 images
  takes ~46 s, so steps 3–4 cost about 5 minutes of wall time.
* CPU: 2 cores, 3.9 GB RAM. The existing U-Net (3 levels, 128×128 crops, 472 k params)
  trained at a measured **99.5 s/epoch** for 75 epochs (~124 min). The hybrid is
  224×224 with a 4-stage decoder and is expected to cost several times that per epoch.
* **This is the one genuine scheduling risk** — see the compute question in the
  accompanying message.
