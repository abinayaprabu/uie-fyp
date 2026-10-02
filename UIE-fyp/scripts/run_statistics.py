#!/usr/bin/env python3
"""Stage A: statistical validation, ranking and selection of the 25 features.

Run from the project root:   python scripts/run_statistics.py

WHAT IT DOES (and what it deliberately does NOT do)
---------------------------------------------------
TRAIN rows only (n = 623) drive every decision:
  descriptives -> univariate regression F-test (SSIM and PSNR) -> BH-FDR
  -> Pearson/Spearman effect sizes -> Random-Forest permutation importance
  (fit on TRAIN, scored on VALIDATION) -> documented combined ranking
  -> |r| >= 0.90 redundancy removal -> final selected set.

The VALIDATION split is used only to SCORE the Random Forest's permutation
importance; it is never trained on. The TEST split is never read by this
script at all: the CSV is filtered to train/val before anything else happens,
and the script records id fingerprints as audit evidence.

Nothing is fixed to 14: the size of the final set is whatever the funnel
produces. The old 14-feature set is loaded only for a documented comparison.

OUTPUTS (all under results/statistics/):
    descriptives_train.csv / descriptives_all.csv
    hypothesis_tests.csv            (F, raw p, BH-adjusted p, Bonferroni, r, rho)
    correlation_matrix_train.csv    (Pearson, 25x25)
    correlation_matrix_spearman_train.csv
    rf_permutation_importance.csv   (both targets + combined ranking)
    redundant_pairs_train.csv       (|r| >= 0.90 removals with reasons)
    selected_features.csv           (THE final set with its full evidence)
    selection_funnel.csv            (25 -> significant -> final, counts + names)
    selection_summary.md            (human-readable; OLD-14 vs NEW comparison)

This script performs no training of any neural network and writes nothing
outside results/statistics/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stats.correlation import (  # noqa: E402
    correlation_matrix, pearson_spearman, spearman_matrix,
)
from stats.descriptive import describe_features  # noqa: E402
from stats.hypothesis_tests import univariate_f_tests  # noqa: E402
from stats.multiple_testing import benjamini_hochberg, bonferroni  # noqa: E402
from stats.selector import combine_rankings, permutation_ranking, select_final_set  # noqa: E402
from cnn.dataset_pairs import ids_fingerprint  # noqa: E402
from src.config import (  # noqa: E402
    CORR_THRESHOLD, FEATURE_NAMES_25, FEATURE_RESULTS_DIR,
    PERM_N_REPEATS, RANDOM_STATE, RESULTS_DIR, RF_MIN_SAMPLES_LEAF,
    RF_N_ESTIMATORS, SPLIT_CSV,
)

QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"
OLD_SELECTION = FEATURE_RESULTS_DIR / "final_selected_features.csv"
OUT_DIR = RESULTS_DIR / "statistics"
ALPHA = 0.05
TARGETS = ("ssim", "psnr")


def load_train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The feature/quality table restricted to the train and val splits.

    The merge is an INNER join on image_name against the frozen split file, so
    a row can only appear if the split file assigns it. Test rows are dropped
    here, before any statistic is computed -- this is the leakage guard.
    """
    df = pd.read_csv(QUALITY_CSV)
    split = pd.read_csv(SPLIT_CSV)
    merged = df.merge(split, on="image_name", how="inner")
    if len(merged) != len(df):
        raise ValueError(f"split file does not cover every row: "
                         f"{len(merged)} of {len(df)}")
    train = merged[merged["split"] == "train"].reset_index(drop=True)
    val = merged[merged["split"] == "val"].reset_index(drop=True)
    if len(train) != 623 or len(val) != 134:
        raise ValueError(f"unexpected split sizes: train {len(train)}, "
                         f"val {len(val)} (expected 623 / 134)")
    # Audit evidence: the exact ids this analysis touched.
    print(f"train ids fingerprint : {ids_fingerprint(train['image_name'].tolist())}")
    print(f"val   ids fingerprint : {ids_fingerprint(val['image_name'].tolist())}")
    n_test = int((merged["split"] == "test").sum())
    print(f"test rows in the loaded label file: {n_test} (never used)")
    print("test rows used for fitting/selection: 0")
    return train, val


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, val = load_train_val()
    features = list(FEATURE_NAMES_25)

    X_train = train[features].to_numpy(dtype=float)
    X_val = val[features].to_numpy(dtype=float)

    # ------------------------------------------------------------------ 1
    # Descriptive statistics (context only -- no decisions are made here).
    desc_train = describe_features(train, features)
    desc_all = describe_features(pd.concat([train, val], ignore_index=True), features)
    desc_train.to_csv(OUT_DIR / "descriptives_train.csv", index=False)
    desc_all.to_csv(OUT_DIR / "descriptives_all.csv", index=False)
    print(f"\n[1] descriptive statistics written ({len(features)} features)")

    # ------------------------------------------------------------------ 2
    # Hypothesis tests: univariate regression F-test per (feature, target),
    # then Benjamini-Hochberg FDR within each target's family of 25 p-values.
    test_rows = []
    adjusted_tables = {}
    for target in TARGETS:
        y_train = train[target].to_numpy(dtype=float)
        tests = univariate_f_tests(X_train, y_train, features)
        p_raw = np.array([t["p_raw"] for t in tests])
        p_bh = benjamini_hochberg(p_raw)
        p_bonf = bonferroni(p_raw)
        assoc = {a["feature"]: a for a in
                 pearson_spearman(X_train, y_train, features)}
        for i, t in enumerate(tests):
            test_rows.append({
                "target": target,
                "feature": t["feature"],
                "F": t["F"],
                "p_raw": t["p_raw"],
                "p_bh_fdr": float(p_bh[i]),
                "p_bonferroni": float(p_bonf[i]),
                "significant_fdr_0.05": bool(p_bh[i] < ALPHA),
                "pearson_r": t["pearson_r"],
                "spearman_rho": assoc[t["feature"]]["spearman_rho"],
            })
        adjusted_tables[target] = pd.DataFrame({
            "feature": features,
            f"adjusted_p_{target}": p_bh,
        })
    tests_df = pd.DataFrame(test_rows)
    tests_df.to_csv(OUT_DIR / "hypothesis_tests.csv", index=False)
    n_sig = tests_df.groupby("target")["significant_fdr_0.05"].sum().to_dict()
    print(f"[2] F-tests + BH-FDR: significant features per target {n_sig} "
          f"of {len(features)}")

    # ------------------------------------------------------------------ 3
    # Correlation structure (association + redundancy input), TRAIN only.
    corr = correlation_matrix(train, features)
    corr.to_csv(OUT_DIR / "correlation_matrix_train.csv")
    spearman_matrix(train, features).to_csv(
        OUT_DIR / "correlation_matrix_spearman_train.csv")
    # documented algebraically-linked pairs, reported for transparency
    algebraic = [("variance", "std", "variance = std^2"),
                 ("ASM", "energy", "ASM = energy^2"),
                 ("rms_contrast", "std", "rms_contrast = std / mean (up to sign)")]
    algebra_rows = []
    for a, b, why in algebraic:
        if a in features and b in features:
            algebra_rows.append({"feature_a": a, "feature_b": b,
                                 "pearson_r": float(corr.loc[a, b]),
                                 "note": why})
    pd.DataFrame(algebra_rows).to_csv(
        OUT_DIR / "algebraically_linked_pairs.csv", index=False)
    print(f"[3] correlation matrices written; "
          f"{int((np.abs(corr.to_numpy()) >= CORR_THRESHOLD).sum() / 2 - len(features) / 2)}"
          f" feature pairs exceed |r| >= {CORR_THRESHOLD}")

    # ------------------------------------------------------------------ 4
    # Predictive importance: Random Forest on TRAIN, permutation on VALIDATION.
    ranking_frames = {}
    for target in TARGETS:
        ranking_frames[target] = permutation_ranking(
            X_train, train[target].to_numpy(dtype=float),
            X_val, val[target].to_numpy(dtype=float),
            features,
            n_estimators=RF_N_ESTIMATORS,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            n_repeats=PERM_N_REPEATS,
            random_state=RANDOM_STATE,
        )
        print(f"[4] RF permutation importance ({target}) on validation: "
              f"top-3 = {ranking_frames[target]['feature'].head(3).tolist()}, "
              f"val R^2 = {ranking_frames[target]['rf_val_r2'].iloc[0]:.4f}")

    combined = combine_rankings(ranking_frames["ssim"], ranking_frames["psnr"])
    combined.to_csv(OUT_DIR / "rf_permutation_importance.csv", index=False)
    print("[4] combined ranking = mean of the two 1-based ranks, ties broken by "
          "mean normalised importance, then alphabetically")

    # ------------------------------------------------------------------ 5
    # The funnel: FDR screen -> combined rank -> redundancy removal.
    adjusted = (adjusted_tables["ssim"]
                .merge(adjusted_tables["psnr"], on="feature"))
    funnel = select_final_set(combined, adjusted, corr,
                              alpha=ALPHA, threshold=CORR_THRESHOLD)
    pd.DataFrame(funnel["removed"]).to_csv(
        OUT_DIR / "redundant_pairs_train.csv", index=False)

    final = funnel["final"]
    final_rows = []
    for rank, name in enumerate(final, start=1):
        row = {"rank": rank, "feature": name}
        sel = combined.loc[combined["feature"] == name].iloc[0]
        for col in ("combined_rank", "rank_ssim", "rank_psnr",
                    "importance_ssim", "importance_psnr"):
            row[col] = float(sel[col])
        for target in TARGETS:
            trow = tests_df[(tests_df["target"] == target)
                            & (tests_df["feature"] == name)].iloc[0]
            row[f"F_{target}"] = float(trow["F"])
            row[f"p_raw_{target}"] = float(trow["p_raw"])
            row[f"p_bh_{target}"] = float(trow["p_bh_fdr"])
            row[f"pearson_r_{target}"] = float(trow["pearson_r"])
            row[f"spearman_rho_{target}"] = float(trow["spearman_rho"])
        final_rows.append(row)
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(OUT_DIR / "selected_features.csv", index=False)

    funnel_df = pd.DataFrame([
        {"stage": "all features", "n": len(features),
         "features": ";".join(funnel["all_features"])},
        {"stage": "significant (BH-FDR < 0.05, SSIM or PSNR)",
         "n": len(funnel["significant"]), "features": ";".join(funnel["significant"])},
        {"stage": "final (after |r| >= 0.90 redundancy removal)", "n": len(final),
         "features": ";".join(final)},
    ])
    funnel_df.to_csv(OUT_DIR / "selection_funnel.csv", index=False)

    # ------------------------------------------------------------------ 6
    # OLD-14 comparison (documentation only -- the old set is NOT used to
    # make any decision in this script).
    old_features: list[str] = []
    if OLD_SELECTION.exists():
        old = pd.read_csv(OLD_SELECTION)
        col = "feature" if "feature" in old.columns else old.columns[0]
        old_features = old[col].astype(str).tolist()
    lines = []
    lines.append("# Stage A — statistical feature selection (TRAIN only)\n")
    lines.append(f"- train rows: {len(train)} (fingerprint "
                 f"`{ids_fingerprint(train['image_name'].tolist())}`)")
    lines.append(f"- validation rows used for permutation importance: {len(val)} "
                 f"(fingerprint `{ids_fingerprint(val['image_name'].tolist())}`)")
    lines.append("- test rows read anywhere in this analysis: 0\n")
    lines.append("## Funnel\n")
    lines.append("| stage | n | features |")
    lines.append("|---|---|---|")
    for _, r in funnel_df.iterrows():
        lines.append(f"| {r['stage']} | {r['n']} | {r['features']} |")
    lines.append("")
    if old_features:
        lines.append("## OLD 14-feature set vs NEW statistically selected set\n")
        lines.append(f"- OLD (RF-permutation + |r|>=0.90, from "
                     f"`results/feature/final_selected_features.csv`): "
                     f"{', '.join(old_features)}")
        lines.append(f"- NEW: {', '.join(final)}")
        only_old = [f for f in old_features if f not in final]
        only_new = [f for f in final if f not in old_features]
        lines.append(f"- dropped relative to OLD: {', '.join(only_old) or '—'}")
        lines.append(f"- added relative to NEW: {', '.join(only_new) or '—'}")
        lines.append("")
    lines.append("## Definitions used\n")
    lines.append("- relevance: univariate regression F-test (H0: slope = 0), "
                 "Benjamini-Hochberg FDR within each target's family of 25 tests, "
                 f"alpha = {ALPHA}")
    lines.append("- effect size: Pearson r (same linear model as the F-test); "
                 "Spearman rho as the monotonic robustness check")
    lines.append("- predictive importance: RandomForestRegressor (train), "
                 "permutation importance on validation")
    lines.append("- combined ranking: (rank_ssim + rank_psnr) / 2, ties broken by "
                 "mean normalised importance then alphabetical")
    lines.append(f"- redundancy: greedy removal while keeping earlier-ranked "
                 f"features, |Pearson r| >= {CORR_THRESHOLD} (train only)")
    (OUT_DIR / "selection_summary.md").write_text("\n".join(lines) + "\n")

    print(f"[5] final selected set ({len(final)}): {', '.join(final)}")
    if old_features:
        print(f"    OLD-14 not in NEW: "
              f"{[f for f in old_features if f not in final]}")
        print(f"    NEW not in OLD-14: "
              f"{[f for f in final if f not in old_features]}")
    print(f"\nAll Stage-A outputs written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
