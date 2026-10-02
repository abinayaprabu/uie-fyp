# Stage A — statistical feature selection (TRAIN only)

- train rows: 623 (fingerprint `02317209a3a5acff`)
- validation rows used for permutation importance: 134 (fingerprint `12a871f98f64b156`)
- test rows read anywhere in this analysis: 0

## Funnel

| stage | n | features |
|---|---|---|
| all features | 25 | red_ratio;dynamic_range;correlation;entropy;homogeneity;mean_blue;keypoint_density;gradient;rms_contrast;mean_red;edge_density;mean_value;mean;glcm_entropy;mean_green;contrast;std;dissimilarity;mean_saturation;laplacian_variance;glcm_variance;colorfulness;ASM;variance;energy |
| significant (BH-FDR < 0.05, SSIM or PSNR) | 17 | red_ratio;dynamic_range;correlation;entropy;mean_blue;rms_contrast;mean_red;mean_value;mean;mean_green;contrast;std;dissimilarity;mean_saturation;laplacian_variance;colorfulness;variance |
| final (after |r| >= 0.90 redundancy removal) | 10 | red_ratio;dynamic_range;correlation;entropy;mean_blue;mean_value;mean;contrast;mean_saturation;variance |

## OLD 14-feature set vs NEW statistically selected set

- OLD (RF-permutation + |r|>=0.90, from `results/feature/final_selected_features.csv`): red_ratio, dynamic_range, entropy, correlation, keypoint_density, homogeneity, mean_blue, mean_value, mean, dissimilarity, laplacian_variance, glcm_variance, mean_saturation, variance
- NEW: red_ratio, dynamic_range, correlation, entropy, mean_blue, mean_value, mean, contrast, mean_saturation, variance
- dropped relative to OLD: keypoint_density, homogeneity, dissimilarity, laplacian_variance, glcm_variance
- added relative to NEW: contrast

## Definitions used

- relevance: univariate regression F-test (H0: slope = 0), Benjamini-Hochberg FDR within each target's family of 25 tests, alpha = 0.05
- effect size: Pearson r (same linear model as the F-test); Spearman rho as the monotonic robustness check
- predictive importance: RandomForestRegressor (train), permutation importance on validation
- combined ranking: (rank_ssim + rank_psnr) / 2, ties broken by mean normalised importance then alphabetical
- redundancy: greedy removal while keeping earlier-ranked features, |Pearson r| >= 0.9 (train only)
