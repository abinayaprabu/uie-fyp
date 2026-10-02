#!/usr/bin/env python3
"""Evaluate top-k feature subsets on VALIDATION data; test stays untouched.

For each k in SUBSET_SIZES (default 15/12/10/8/6 — clipped to the number of
post-correlation survivors): take the top-k rank-ordered survivors, fit one
RandomForest per target on TRAIN, evaluate on VAL with R²/RMSE/MAE, and
compute the project-defined combined score = average val R² across SSIM/PSNR.

The best-k by that combined score is written to final_selected_features.csv.
TEST metrics are NOT computed here; the RF-on-final-features baseline is
evaluated on test exactly once, later, in scripts/run_baselines.py together
with the CNN comparisons (single joint final evaluation — see README).

Outputs (results/feature/):
  feature_subset_evaluation.csv, final_selected_features.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    FEATURE_RESULTS_DIR,
    RANDOM_STATE,
    RF_MIN_SAMPLES_LEAF,
    RF_N_ESTIMATORS,
    SPLIT_CSV,
    SUBSET_SIZES,
)
from src.metrics import combined_r2, regression_metrics  # noqa: E402

QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"


def main() -> int:
    df = pd.read_csv(QUALITY_CSV).merge(pd.read_csv(SPLIT_CSV), on="image_name")
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    survivors = pd.read_csv(FEATURE_RESULTS_DIR / "selected_features.csv"
                            ).sort_values("rank")["feature"].tolist()
    print(f"Survivors after correlation filter: {len(survivors)}")

    # Dedupe: grid values above the survivor count clip to the same k.
    grid = sorted({min(k, len(survivors)) for k in SUBSET_SIZES}, reverse=True)
    print(f"Subset grid {tuple(SUBSET_SIZES)} clipped to survivors: {grid}")
    rows = []
    for k in grid:
        feats = survivors[:k]
        res: dict = {"k": k, "features": ",".join(feats)}
        for target in ("ssim", "psnr"):
            model = RandomForestRegressor(
                n_estimators=RF_N_ESTIMATORS, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE, n_jobs=-1)
            model.fit(train[feats], train[target])
            m = regression_metrics(val[target], model.predict(val[feats]))
            for key in ("r2", "rmse", "mae", "pearson_r"):
                res[f"{target}_{key}"] = round(m[key], 4)
        res["avg_R2_SSIM_PSNR_project_defined"] = round(
            combined_r2(res["ssim_r2"], res["psnr_r2"]), 4)
        rows.append(res)
        print(f"k={k:2d}: SSIM R²={res['ssim_r2']:.4f} PSNR R²={res['psnr_r2']:.4f} "
              f"avg={res['avg_R2_SSIM_PSNR_project_defined']:.4f}")

    ev = pd.DataFrame(rows)
    ev.to_csv(FEATURE_RESULTS_DIR / "feature_subset_evaluation.csv", index=False)
    best_k = int(ev.loc[ev["avg_R2_SSIM_PSNR_project_defined"].idxmax(), "k"])
    final = survivors[:best_k]
    pd.DataFrame({"rank": range(1, best_k + 1), "feature": final}
                 ).to_csv(FEATURE_RESULTS_DIR / "final_selected_features.csv", index=False)
    print(f"\nBest subset by project-defined avg val R²: k={best_k}")
    for i, f in enumerate(final, 1):
        print(f"  {i}. {f}")
    print("CHECK 13 (test set not used for selection): PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
