#!/usr/bin/env python3
"""Train + evaluate all baselines and the proposed model on the TEST split.

Models (all use train for fitting, val for early stopping / hyperparameters
fixed a priori, test exactly ONCE here):
  baseline_rf    : RandomForest on final features (per target, params fixed)
  mlp_final      : torch MLP on final features (cnn.train --model mlp)
  image_only     : torch CNN on image only     (cnn.train --model image_only)
  hybrid_final   : PROPOSED torch hybrid CNN + final features

The RF baseline is trained here; neural models are trained via cnn.train and
evaluated via cnn.evaluate (single joint test evaluation for every model).
Results are consolidated into results/comparison/baseline_metrics.csv.

Skip flags allow re-running evaluation without retraining neural nets:
    --skip-train   (evaluate existing checkpoints only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn.evaluate import evaluate_run  # noqa: E402
from cnn.train import train_model  # noqa: E402
from src.config import (  # noqa: E402
    COMPARISON_RESULTS_DIR,
    FEATURE_RESULTS_DIR,
    RANDOM_STATE,
    RF_MIN_SAMPLES_LEAF,
    RF_N_ESTIMATORS,
    SPLIT_CSV,
)
from src.metrics import (  # noqa: E402
    bootstrap_r2_ci, combined_r2, regression_metrics,
)

QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"
NEURAL_RUNS = [("mlp", "final"), ("image_only", "nofeat"), ("hybrid", "final")]


def rf_baseline(feat_names: list[str]) -> dict:
    df = pd.read_csv(QUALITY_CSV).merge(pd.read_csv(SPLIT_CSV), on="image_name")
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    print(f"RF baseline on {len(feat_names)} final features "
          f"(train={len(train)}, test={len(test)})")
    row: dict = {"model": "baseline_rf", "features": ",".join(feat_names),
                 "n_test": len(test)}
    test_out = COMPARISON_RESULTS_DIR / "rf_baseline_test_predictions.csv"
    COMPARISON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds = {"image_name": test["image_name"].tolist(),
             "ssim_true": test["ssim"].tolist(), "psnr_true": test["psnr"].tolist()}
    # L7 FIX: each forest is now fit ONCE per target and reused for both the
    # metrics row and the predictions file. The previous version fit it twice;
    # with a fixed random_state the two fits agreed, so this is purely a 2x
    # saving, not a behaviour change.
    for target in ("ssim", "psnr"):
        m = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS,
                                  min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                                  random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(train[feat_names], train[target])
        p = m.predict(test[feat_names])
        preds[f"{target}_pred"] = p
        met = regression_metrics(test[target], p)
        for k in ("r2", "rmse", "mae", "pearson_r"):
            row[f"{target}_{k}"] = round(met[k], 4)
        # H5: a point R2 on 133 test images is not interpretable without an
        # interval; report the percentile bootstrap CI alongside it.
        lo, hi = bootstrap_r2_ci(test[target], p)
        row[f"{target}_r2_ci95_lo"] = round(lo, 4)
        row[f"{target}_r2_ci95_hi"] = round(hi, 4)
    row["avg_R2_SSIM_PSNR_project_defined"] = round(
        combined_r2(row["ssim_r2"], row["psnr_r2"]), 4)
    pd.DataFrame(preds).to_csv(test_out, index=False)
    print(f"  RF: SSIM R²={row['ssim_r2']:.4f} PSNR R²={row['psnr_r2']:.4f} "
          f"avg={row['avg_R2_SSIM_PSNR_project_defined']:.4f}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    COMPARISON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feat_names = (pd.read_csv(FEATURE_RESULTS_DIR / "final_selected_features.csv")
                  .sort_values("rank")["feature"].tolist())
    rows = [rf_baseline(feat_names)]

    for model_name, feat_tag in NEURAL_RUNS:
        run_tag = f"{model_name}_{feat_tag}"
        if not args.skip_train:
            # train_model() derives the same run_tag internally; image_only
            # ignores the features argument.
            train_model(model_name, "final")
        met = evaluate_run(run_tag)
        rows.append({
            "model": run_tag, "features": ",".join(met["features"]),
            "n_test": met["n_test"],
            **{f"{t}_{k}": round(met[t][k], 4)
               for t in ("ssim", "psnr") for k in ("r2", "rmse", "mae", "pearson_r")},
            "avg_R2_SSIM_PSNR_project_defined": round(
                met["avg_R2_SSIM_PSNR_project_defined"], 4),
        })

    out = COMPARISON_RESULTS_DIR / "baseline_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")
    print(pd.DataFrame(rows)[["model", "ssim_r2", "psnr_r2",
                              "avg_R2_SSIM_PSNR_project_defined"]].to_string(index=False))
    print("\nHonest-reporting rule: if hybrid_final does not beat the baselines, "
          "that is the finding — do not re-tune on test to force a win.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
