#!/usr/bin/env python3
"""Leakage-free feature ranking (CHECK 13 enforced).

Methodology (per target: SSIM, PSNR):
  1. Fit RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
     random_state=42, n_jobs=-1) on the TRAIN split only (25 features).
  2. Compute PERMUTATION importance on the VALIDATION split (held-out data,
     scoring=neg_mean_squared_error, n_repeats=5). Permutation importance is
     NOT the same as the forest's impurity (MDI) importance: it measures the
     drop in held-out performance when a feature is shuffled.
  3. Rank features by mean permutation importance per target, then combine:
     combined = (norm_ssim + norm_psnr) / 2 after min-max normalising each
     target's importances to [0, 1].

The TEST split is never touched here (CHECK 13). Any exploratory ranking that
used all 890 images for selection would be optimistic for generalisation;
that is why this script re-runs selection inside the train split.

Outputs (results/feature/):
  ranking_ssim.csv, ranking_psnr.csv, ranking_combined.csv
  (columns: rank, feature, importance_mean, importance_std)
and prints held-out (val) R²/RMSE/MAE for the two forests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    FEATURE_NAMES_25,
    FEATURE_RESULTS_DIR,
    PERM_N_REPEATS,
    RANDOM_STATE,
    RF_MIN_SAMPLES_LEAF,
    RF_N_ESTIMATORS,
    SPLIT_CSV,
)
from src.metrics import combined_r2, regression_metrics  # noqa: E402

QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"


def load_split_frames():
    df = pd.read_csv(QUALITY_CSV)
    split = pd.read_csv(SPLIT_CSV)
    merged = df.merge(split, on="image_name", how="inner")
    assert len(merged) == len(df), "split file does not cover all feature rows!"
    # CHECK 13 evidence: test ids are present but explicitly excluded below.
    train = merged[merged["split"] == "train"].reset_index(drop=True)
    val = merged[merged["split"] == "val"].reset_index(drop=True)
    test = merged[merged["split"] == "test"].reset_index(drop=True)
    assert not set(train["image_name"]) & set(test["image_name"])
    print(f"train={len(train)} val={len(val)} test={len(test)} (test untouched here)")
    return train, val


def rank_target(train: pd.DataFrame, val: pd.DataFrame, target: str) -> pd.DataFrame:
    X_train, y_train = train[FEATURE_NAMES_25], train[target]
    X_val, y_val = val[FEATURE_NAMES_25], val[target]
    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    m = regression_metrics(y_val, model.predict(X_val))
    print(f"[{target}] val R²={m['r2']:.4f} RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} "
          f"Pearson r={m['pearson_r']:.4f}")
    perm = permutation_importance(
        model, X_val, y_val, n_repeats=PERM_N_REPEATS,
        random_state=RANDOM_STATE, n_jobs=-1, scoring="neg_mean_squared_error")
    ranking = pd.DataFrame({
        "feature": FEATURE_NAMES_25,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    return ranking, m


def main() -> int:
    FEATURE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train, val = load_split_frames()
    r_ssim, m_ssim = rank_target(train, val, "ssim")
    r_psnr, m_psnr = rank_target(train, val, "psnr")

    r_ssim.to_csv(FEATURE_RESULTS_DIR / "ranking_ssim.csv", index=False)
    r_psnr.to_csv(FEATURE_RESULTS_DIR / "ranking_psnr.csv", index=False)

    def minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else s * 0.0

    combined = pd.DataFrame({
        "feature": FEATURE_NAMES_25,
        "importance_ssim": minmax(r_ssim.set_index("feature")["importance_mean"]),
        "importance_psnr": minmax(r_psnr.set_index("feature")["importance_mean"]),
    })
    combined["importance_combined"] = (
        combined["importance_ssim"] + combined["importance_psnr"]) / 2.0
    combined = combined.sort_values("importance_combined", ascending=False
                                    ).reset_index(drop=True)
    combined.insert(0, "rank", combined.index + 1)
    combined.to_csv(FEATURE_RESULTS_DIR / "ranking_combined.csv", index=False)

    print(f"\nAverage held-out (val) R² across SSIM and PSNR (project-defined): "
          f"{combined_r2(m_ssim['r2'], m_psnr['r2']):.4f}")
    print("\nCombined ranking (train-fit, val permutation importance):")
    for _, r in combined.iterrows():
        print(f"  {int(r['rank']):2d}. {r['feature']:20s} {r['importance_combined']:.4f}")
    print("\nCHECK 13 (test set not used for selection): PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
