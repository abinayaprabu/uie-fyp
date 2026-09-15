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
> model input.

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
- **Option A (MLP on 8 features)** is a baseline, and is called an MLP — never a CNN.
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

## 11. Final selected features
Determined by the leakage-free run (see `results/feature/`); the historically
reported 8 (`edge_density, energy, std, red_ratio, entropy, mean_blue,
keypo
...[truncated 4045 chars]