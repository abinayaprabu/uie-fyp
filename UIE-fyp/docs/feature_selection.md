# Feature selection — how the 25 features become the model's guide

*Written 2026-10-02. The numbers live in `results/feature_selection/`
(auto-generated); this document explains the method and the rules. Nothing here
recomputes anything.*

## 1. Why select at all

The 25 handcrafted features are highly correlated with each other
(e.g. `variance = std²`, `ASM = energy²`) and the encoder only needs a compact,
defensible conditioning signal. Feeding all 25 to the MLP would work, but it
would not answer the project's question *"which image characteristics actually
carry information about enhancement quality?"* — which is the contribution of
this stage.

## 2. The exact procedure (locked)

```
25 features  ──TRAIN only──▶  univariate f_regression (vs SSIM, vs PSNR)
                              │  F, raw p, Pearson r (effect size), Spearman ρ
                              ▼
                         Benjamini-Hochberg FDR (α = 0.05) + Bonferroni
                              │  keep features significant in SSIM OR PSNR
                              ▼
                 Random Forest permutation importance (5 repeats)
                 fitted on TRAIN, scored on VALIDATION  (never test)
                              │  separate SSIM / PSNR rankings
                              ▼
                       combined rank = mean of the two 1-based ranks
                              ▼
                Pearson redundancy filtering on TRAIN, |r| ≥ 0.90
                              │  keep the higher-ranked feature of each pair
                              ▼
                         FINAL SELECTED FEATURES
```

Implementation: `stats/` (descriptive → hypothesis_tests → multiple_testing →
correlation → redundancy → selector), driver `scripts/run_statistics.py`,
packaging script `scripts/make_feature_selection_outputs.py`.

## 3. What each method answers (the viva table)

| Step | Question it answers | Why this method |
|---|---|---|
| `f_regression` | Is there a linear association between this feature and enhancement quality? | Targets (SSIM, PSNR) are continuous; the F-test is the standard univariate regression test |
| Pearson r | How large is that association? | Effect size of the same model — not independent evidence |
| Spearman ρ | Is the association monotone even if not linear? | Robustness check |
| BH-FDR | Which associations survive multiple testing? | 50 tests; controls false discoveries with more power than Bonferroni |
| RF permutation importance | Does a flexible model actually rely on the feature when predicting on unseen (validation) data? | Catches non-linear effects the F-test misses |
| Combined rank | One ordering over two targets | Mean of the two 1-based ranks; documented tie-breaks |
| Pearson redundancy (train) | Do two features carry the same information? | |r| ≥ 0.90 keeps the stronger one |

## 4. Results (measured)

- 17 of 25 features are significant after BH-FDR in at least one target family
  (SSIM 14, PSNR 17).
- 19 train pairs exceed |r| = 0.90.
- Final set: **10 features** — `red_ratio, dynamic_range, correlation, entropy,
  mean_blue, mean_value, mean, contrast, mean_saturation, variance`
  (full tables: `results/feature_selection/feature_selection_report.md`,
  `statistical_results.csv/.xlsx`, `feature_ranking.csv`).
- The OLD 14-feature set is preserved untouched at
  `results/feature/final_selected_features.csv`; the five features it has that
  the new set does not (`keypoint_density, homogeneity, dissimilarity,
  laplacian_variance, glcm_variance`) were each removed by the train-only
  0.90 redundancy rule against a stronger-ranked survivor. The new set adds
  `contrast`.

The specification explicitly forbids forcing the number 14: the
statistically justified set is used, and the OLD set is still reported for
comparison.

## 5. Leakage rules (non-negotiable)

1. All association statistics and the correlation matrix use **train rows
   only**.
2. The RF is **fitted on train**, permutation importance is scored on
   **validation**. Test rows are never read.
3. The feature scaler (`sklearn.StandardScaler`) is fitted on **train only**
   and stored with the checkpoint; `scripts/leakage_audit.py` re-checks it.
4. `scripts/run_statistics.py` asserts the split sizes and prints the id
   fingerprints; `scripts/leakage_audit.py` recomputes the whole funnel and
   requires the stored selected set to be reproduced exactly.

## 6. How to reproduce

```bash
python scripts/run_statistics.py                 # writes results/statistics/
python scripts/make_feature_selection_outputs.py # writes results/feature_selection/
python scripts/leakage_audit.py                  # 12 checks, must pass
```

## 7. What this stage does NOT claim

- It does not claim these features *cause* better enhancement — that is the
  job of the image-only vs feature-guided ablation.
- It does not claim neural-network importance; the tests are statistical and
  predictive, on train/validation only.
- It does not claim the set is optimal in any absolute sense; it is the set
  produced by the pre-registered procedure, with the OLD set reported.
