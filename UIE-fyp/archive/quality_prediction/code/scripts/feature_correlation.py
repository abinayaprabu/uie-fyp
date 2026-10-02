#!/usr/bin/env python3
"""Pearson redundancy filter on TRAIN features only (CHECK 13 enforced).

Method:
  1. Compute the 25x25 Pearson correlation matrix of handcrafted features
     using TRAIN-split rows only (never val/test).
  2. List all pairs with abs(r) >= 0.90 as highly correlated.
  3. Greedy selection in combined-rank order: walk the ranking from best to
     worst; keep a feature unless it correlates >= 0.90 (absolute) with an
     already-kept, higher-ranked feature. This stage removes
     feature-to-feature redundancy only — correlation with the TARGET is never
     a removal criterion.

Outputs (results/feature/):
  feature_correlation_matrix.csv, highly_correlated_features.csv,
  selected_features.csv (rank-ordered survivors), removed_redundant_features.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    CORR_THRESHOLD,
    FEATURE_NAMES_25,
    FEATURE_RESULTS_DIR,
    SPLIT_CSV,
)

QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"


def main() -> int:
    df = pd.read_csv(QUALITY_CSV).merge(pd.read_csv(SPLIT_CSV), on="image_name")
    train = df[df["split"] == "train"].reset_index(drop=True)
    print(f"Correlation computed on TRAIN rows only: n={len(train)}")

    corr = train[FEATURE_NAMES_25].corr(method="pearson")
    corr.to_csv(FEATURE_RESULTS_DIR / "feature_correlation_matrix.csv")

    pairs = []
    for i, a in enumerate(FEATURE_NAMES_25):
        for b in FEATURE_NAMES_25[i + 1:]:
            r = float(corr.loc[a, b])
            if abs(r) >= CORR_THRESHOLD:
                pairs.append({"feature_a": a, "feature_b": b, "pearson_r": r,
                              "abs_r": abs(r)})
    pairs_df = pd.DataFrame(pairs).sort_values("abs_r", ascending=False)
    pairs_df.to_csv(FEATURE_RESULTS_DIR / "highly_correlated_features.csv", index=False)
    print(f"Highly correlated pairs (|r| >= {CORR_THRESHOLD}): {len(pairs_df)}")
    for _, p in pairs_df.iterrows():
        print(f"  {p['feature_a']:20s} <-> {p['feature_b']:20s} r={p['pearson_r']:+.4f}")

    ranking = pd.read_csv(FEATURE_RESULTS_DIR / "ranking_combined.csv")
    ordered = ranking.sort_values("rank")["feature"].tolist()
    kept: list[str] = []
    removed: list[dict] = []
    for feat in ordered:
        blocker = next((k for k in kept
                        if abs(float(corr.loc[feat, k])) >= CORR_THRESHOLD), None)
        if blocker is None:
            kept.append(feat)
        else:
            removed.append({"removed_feature": feat, "kept_feature": blocker,
                            "pearson_r": float(corr.loc[feat, blocker])})

    pd.DataFrame({"rank": range(1, len(kept) + 1), "feature": kept}
                 ).to_csv(FEATURE_RESULTS_DIR / "selected_features.csv", index=False)
    pd.DataFrame(removed).to_csv(
        FEATURE_RESULTS_DIR / "removed_redundant_features.csv", index=False)
    print(f"\nSelected after redundancy removal: {len(kept)}")
    for i, f in enumerate(kept, 1):
        print(f"  {i:2d}. {f}")
    print("Removed:")
    for r in removed:
        print(f"  {r['removed_feature']:20s} -> kept {r['kept_feature']:20s} "
              f"(r={r['pearson_r']:+.4f})")
    print("CHECK 13 (test set not used for selection): PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
