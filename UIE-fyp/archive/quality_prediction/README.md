# Archive — the discarded quality-prediction direction

**Second cleanup, 2026-10-02:** the archived *code* and *figures* have now been
deleted from the working tree (they are preserved in git history, commit
`383206e`, and this README records exactly what they were and how to restore
them). The small **result tables are kept** because the final report cites them
as prior work, and an examiner must be able to check those numbers.

What this folder still contains (KEEP — evidence cited in the report):

    results/cnn/<run>/{metrics.json,test_predictions.csv,train_history.csv}
        four quality-prediction runs: mlp_final, image_only_nofeat,
        hybrid_all25, hybrid_final
    results/comparison/baseline_metrics.csv
        consolidated test R²/RMSE for every prior-work model
    results/comparison/ablation_results.csv
        the prediction-side ablation (image-only vs feature-fused predictor)
    results/comparison/rf_baseline_test_predictions.csv
        Random-Forest baseline per-image test predictions

What was deleted in the second cleanup (restorable from git):

    code/cnn/{model,train,evaluate,predict}.py
    code/scripts/{run_baselines,run_ablation,feature_ranking,
                  feature_subset_evaluation,feature_correlation,make_plots,
                  verify_results}.py
    plots/{model_comparison,predicted_vs_actual_ssim,predicted_vs_actual_psnr,
           training_curve_mlp_final,training_curve_hybrid_all25,
           training_curve_hybrid_final,training_curve_image_only_nofeat,
           ablation_comparison}.png

Why they could be deleted safely:

1. No file in the final pipeline imports or reads them (checked with
   `git grep` before deleting).
2. They are preserved byte-for-byte in git history; restore with:
   `git checkout 383206e -- UIE-fyp/archive/quality_prediction/code UIE-fyp/archive/quality_prediction/plots`
3. The final report only needs the *numbers*, which are in the kept CSVs; the
   methodology is described in `docs/audit-and-build-report.md` and
   `docs/feasibility-feature-only-enhancement.md`.
4. Their figures belonged to the rejected direction and could be mistaken for
   final results; the final qualitative figures are in
   `results/metrics/comparison_images/`.

Why the direction was replaced (the negative result that motivated the final
architecture): predicting scalar quality from the image / features / both gave
image-only R² 0.4659 vs hybrid_final 0.4658 on SSIM (within ±0.15 bootstrap
CIs) — i.e. the feature branch added nothing *to prediction*; and the
feature-only enhancement study showed the features alone cannot place spatial
structure. The final project therefore uses the features to CONDITION a CNN
that reconstructs the enhanced image, which is what this repository now does.

Note: the predictors' trained checkpoints were never committed and their
`models/*.pt` files no longer exist; the metrics and per-image predictions
above are the surviving evidence.
