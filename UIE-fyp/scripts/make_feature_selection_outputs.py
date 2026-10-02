#!/usr/bin/env python3
"""Build the clean `results/feature_selection/` deliverable from Stage-A files.

The statistics themselves are produced by `scripts/run_statistics.py` (single
source of truth, TRAIN + VALIDATION rows only).  This script only *packages*
them into the file names the project specification asks for, so nothing is
recomputed and no number can drift:

    results/feature_selection/
        statistical_results.csv     per feature x target: F, raw p, BH-FDR q,
                                    Bonferroni, Pearson r, Spearman rho, status
        statistical_results.xlsx    the same, plus a funnel sheet
        feature_ranking.csv         RF permutation importance + combined rank
        selected_features.txt       the final statistically justified set
        correlation_matrix.csv      TRAIN-only Pearson matrix (25 x 25)
        feature_selection_report.md human-readable report with the exact criteria

Run:  python scripts/make_feature_selection_outputs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    CORR_THRESHOLD, FEATURE_RESULTS_DIR, PERM_N_REPEATS, RESULTS_DIR,
    RF_MIN_SAMPLES_LEAF, RF_N_ESTIMATORS,
)

STATS_DIR = RESULTS_DIR / "statistics"
OUT_DIR = RESULTS_DIR / "feature_selection"
OLD_SELECTION = FEATURE_RESULTS_DIR / "final_selected_features.csv"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hyp = pd.read_csv(STATS_DIR / "hypothesis_tests.csv")
    ranking = pd.read_csv(STATS_DIR / "rf_permutation_importance.csv")
    funnel = pd.read_csv(STATS_DIR / "selection_funnel.csv")
    redundant = pd.read_csv(STATS_DIR / "redundant_pairs_train.csv")
    selected = pd.read_csv(STATS_DIR / "selected_features.csv")
    corr = pd.read_csv(STATS_DIR / "correlation_matrix_train.csv", index_col=0)
    old = pd.read_csv(OLD_SELECTION)

    selected_names = selected["feature"].tolist()
    removed_map = dict(zip(redundant["removed_feature"], redundant["reason"]))

    # ---- statistical_results.csv -----------------------------------------
    stat = hyp.copy()
    stat["n_train"] = 623
    stat["selected"] = stat["feature"].isin(selected_names)
    stat["dropped_as_redundant_with"] = stat["feature"].map(
        dict(zip(redundant["removed_feature"], redundant["kept_feature"])))
    stat["removal_reason"] = stat["feature"].map(removed_map)
    stat = stat.sort_values(["target", "p_bh_fdr"]).reset_index(drop=True)
    stat.to_csv(OUT_DIR / "statistical_results.csv", index=False)

    # ---- statistical_results.xlsx (one sheet per target + funnel) --------
    with pd.ExcelWriter(OUT_DIR / "statistical_results.xlsx") as xl:
        for target in stat["target"].unique():
            stat[stat["target"] == target].to_excel(
                xl, sheet_name=f"f_regression_{target}", index=False)
        ranking.to_excel(xl, sheet_name="rf_permutation_importance", index=False)
        funnel.to_excel(xl, sheet_name="selection_funnel", index=False)
        selected.to_excel(xl, sheet_name="final_selected", index=False)

    # ---- feature_ranking.csv ---------------------------------------------
    ranking.to_csv(OUT_DIR / "feature_ranking.csv", index=False)

    # ---- selected_features.txt -------------------------------------------
    lines = ["# Final selected handcrafted features",
             "# Produced by scripts/run_statistics.py (TRAIN for tests, "
             "VALIDATION for permutation importance; TEST never read).",
             f"# n_features = {len(selected_names)}",
             ""]
    for i, name in enumerate(selected_names, 1):
        lines.append(f"{i}. {name}")
    (OUT_DIR / "selected_features.txt").write_text("\n".join(lines) + "\n")

    # ---- correlation_matrix.csv ------------------------------------------
    corr.to_csv(OUT_DIR / "correlation_matrix.csv")

    # ---- feature_selection_report.md -------------------------------------
    def fmt_feature_table(df: pd.DataFrame) -> str:
        rows = ["| feature | F | raw p | BH-FDR q | Pearson r | Spearman rho | kept |",
                "|---|---|---|---|---|---|---|"]
        for _, r in df.iterrows():
            rows.append(
                f"| {r['feature']} | {r['F']:.2f} | {r['p_raw']:.2e} | "
                f"{r['p_bh_fdr']:.2e} | {r['pearson_r']:+.3f} | "
                f"{r['spearman_rho']:+.3f} | {'yes' if r['selected'] else 'no'} |")
        return "\n".join(rows)

    ssim_tab = fmt_feature_table(stat[stat["target"] == "ssim"])
    psnr_tab = fmt_feature_table(stat[stat["target"] == "psnr"])

    red_lines = ["| removed | kept (higher ranked) | Pearson r | reason |", "|---|---|---|---|"]
    for _, r in redundant.iterrows():
        red_lines.append(f"| {r['removed_feature']} | {r['kept_feature']} | "
                         f"{r['pearson_r']:+.3f} | {r['reason']} |")
    red_tab = "\n".join(red_lines)

    old_names = old["feature"].tolist()
    report = f"""# Feature selection report — 25 handcrafted features to {len(selected_names)}

*Generated by `scripts/make_feature_selection_outputs.py` from the Stage-A outputs
of `scripts/run_statistics.py`. The split is frozen at train 623 / val 134 /
test 133 (seed 42). Every statistic below uses TRAIN rows only, except the
random-forest permutation importance, which is fitted on TRAIN and scored on
VALIDATION. **The test set was never read in this analysis** (asserted in code
and re-checked by `scripts/leakage_audit.py`).*

## 1. Inputs

- 25 handcrafted features extracted from the FULL-resolution preprocessed image
  (`src/features.py`), never from the reference image.
- Targets: SSIM and PSNR of the frozen classical pipeline against the UIEB
  reference (`results/feature/feature_quality_dataset.csv`).

## 2. Test 1 — univariate regression F-test (`f_regression`)

For each feature and each target, H0: the linear-regression slope of the target
on that single feature is zero. F, the raw p-value, the Benjamini-Hochberg
FDR-adjusted q-value (α = 0.05, family = the 25 features of that target) and the
Bonferroni value are reported. Pearson r is the effect size of the SAME linear
model — it is not independent evidence.

### Target SSIM

{ssim_tab}

### Target PSNR

{psnr_tab}

## 3. Test 2 — Random Forest permutation importance

`RandomForestRegressor(n_estimators={RF_N_ESTIMATORS}, min_samples_leaf={RF_MIN_SAMPLES_LEAF}, random_state=42)`
is fitted on TRAIN and scored on VALIDATION with permutation importance ({PERM_N_REPEATS} repeats).
SSIM and PSNR are ranked separately; `combined_rank` is the mean of the two
1-based ranks (ties broken by mean normalised importance, then alphabetically).

Validation R²: SSIM {ranking['rf_val_r2_ssim'].iloc[0]:.4f},
PSNR {ranking['rf_val_r2_psnr'].iloc[0]:.4f}. Full table in
`feature_ranking.csv`.

## 4. Test 3 — redundancy filtering (feature-feature, TRAIN only)

Pearson correlation between features on TRAIN, threshold |r| ≥ {CORR_THRESHOLD}. The
higher-ranked feature is kept. This is redundancy analysis, NOT a
target-correlation test.

{red_tab}

Algebraically linked pairs that the redundancy rule also catches are recorded
separately in `results/statistics/algebraically_linked_pairs.csv`
(e.g. variance = std², ASM = energy²).

## 5. Final selected set — {len(selected_names)} features

**{', '.join(selected_names)}**

Criteria, in this exact order: (1) significant after BH-FDR in the SSIM family
OR the PSNR family; (2) ranked by combined RF permutation importance; (3) no
remaining pair with |Pearson r| ≥ 0.90 on TRAIN (keep the stronger).

The selection was NOT forced to 14 features. The pipeline produced
{len(selected_names)} and the statistically justified set is kept, as the
specification requires.

## 6. OLD 14-feature set vs the statistically justified set

- OLD (previous procedure, preserved at `results/feature/final_selected_features.csv`):
  {', '.join(old_names)}
- NEW: {', '.join(selected_names)}
- dropped relative to OLD: {', '.join(sorted(set(old_names) - set(selected_names)))}
- added relative to NEW: {', '.join(sorted(set(selected_names) - set(old_names)))}

The five dropped features were each removed by the train-only |r| ≥ 0.90 rule
against a stronger-ranked survivor (see the table in section 4). The OLD list is
kept, untouched, so both can be compared in the report.

## 7. What this report does NOT claim

- No claim that these features cause better enhancement. That is what the
  image-only vs feature-guided ablation measures (`results/metrics/`).
- No claim of neural-network importance: the F-test and the permutation
  importance are statistical/predictive measures on TRAIN/VALIDATION only.
- No claim about the test split, which is untouched until final evaluation.
"""
    (OUT_DIR / "feature_selection_report.md").write_text(report)

    print(f"Wrote 6 files to {OUT_DIR}:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size:,} bytes)")
    print(f"\nselected ({len(selected_names)}): {', '.join(selected_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
