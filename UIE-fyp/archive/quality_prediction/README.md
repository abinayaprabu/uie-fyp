# Archive — the discarded quality-prediction direction (2026-10-02)

Everything in this folder belonged to the **previous** project direction:
predicting a scalar quality score (SSIM / PSNR) from an image, from the 25
handcrafted features, or from both. That direction was superseded when the
final architecture was locked: **feature-guided enhancement**, where the
handcrafted features condition a CNN that reconstructs the enhanced image.

Nothing here is deleted, and nothing here is imported by the final pipeline.
The files were *moved* (git history preserved) so the main tree contains only
the final deliverable, while the old work stays available for the report and
for answering "what did you reject and why?" in the viva.

Contents
--------

    code/cnn/model.py            FeatMLP / ImageOnlyCNN / HybridCNN (quality predictors)
    code/cnn/train.py            trainer for the predictors
    code/cnn/evaluate.py         test-set evaluation for the predictors
    code/cnn/predict.py          single-image prediction CLI
    code/scripts/run_baselines.py        trains + evaluates all four predictors
    code/scripts/run_ablation.py         image-only predictor vs hybrid predictor
    code/scripts/feature_ranking.py      RF permutation ranking on all 890 rows
                                         (superseded by stats/selector.py, which is
                                          train/val only and leakage-audited)
    code/scripts/feature_subset_evaluation.py  feature-subset search for the predictors
    code/scripts/feature_correlation.py  correlation study on all rows (superseded)
    code/scripts/make_plots.py           prediction-track plots
    code/scripts/verify_results.py       independent verification of the predictors
    results/cnn/                 metrics.json / test_predictions.csv / train_history.csv
    results/comparison/          baseline_metrics.csv, ablation_results.csv
    plots/                       training curves, predicted-vs-actual scatters,
                                 model comparison, predictor ablation figure

Why each was classified ARCHIVE and not DELETE
-----------------------------------------------
- They document several months of work and the negative result that motivated
  the final design (feature-only and prediction-side feature fusion did **not**
  outperform the image-only predictor: image_only R² 0.4659 vs hybrid_final
  0.4658 on SSIM, within ±0.15 bootstrap CIs).
- `results/comparison/*.csv` are the tables the final report cites as *prior
  work*, and the audit trail for the claims about those models.
- Deleting them would remove the ability to answer the examiners' obvious
  question: "why is your model not just the previous one?"

Important notes
---------------
- Code here still refers to its original paths (`results/cnn/`,
  `results/comparison/`, `cnn/model.py`). They will not run from this folder
  without adjusting those paths; they are kept as a record, not as a supported
  entry point.
- The predictors' trained checkpoints (`models/*.pt`) are **not** in the
  repository (they were never committed and were lost in a sandbox reset).
  Their metrics and per-image predictions are, which is what the report quotes.
- The old documentation (`docs/project-flow.md`, `docs/plain-language-explanation.md`,
  `docs/methodology-audit.md`, `docs/exact-flow.md`) was written when these
  files lived in `cnn/` and `scripts/`; the paths in those documents refer to
  the pre-2026-10-02 layout.
