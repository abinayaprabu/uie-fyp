"""Predictive feature importance (Random Forest) and the assembled selection funnel.

TWO DIFFERENT QUESTIONS, TWO DIFFERENT TOOLS (this distinction is the point)
--------------------------------------------------------------------------
* ``hypothesis_tests.py`` asks: "is there a detectable LINEAR association
  between this feature and quality?"  -> F-test + FDR. Statistical relevance.
* this module asks: "does a flexible model actually USE this feature to
  predict quality?"  -> RandomForestRegressor + PERMUTATION importance.
  Predictive relevance. A feature can pass one and fail the other, and the
  report shows both instead of treating them as the same thing.

PROTOCOL FOR THE RANDOM FOREST (leakage-free)
---------------------------------------------
* fit  : TRAIN rows only (623 images), random_state = 42, 200 trees,
         min_samples_leaf = 2 (from ``src/config.py``: RF_* constants)
* score: PERMUTATION importance on the VALIDATION split (134 images) --
         each feature column is shuffled ``n_repeats`` times and the resulting
         drop in R^2 is recorded. Validation is never trained on, and the TEST
         split is not touched anywhere in this package.
* ranked: separately for SSIM and for PSNR, then combined with the documented
  formula below.

COMBINED RANKING FORMULA (exact definition, no hand-waving)
-----------------------------------------------------------
    rank_ssim   = 1-based rank of the feature by permutation importance for SSIM
    rank_psnr   = 1-based rank of the feature by permutation importance for PSNR
    combined    = (rank_ssim + rank_psnr) / 2        (lower is better)
    tie-break   = higher mean of the two normalised importance columns, then
                  alphabetical feature name (deterministic)

Then the funnel is:

    25 features
      -> (1) FDR screen: keep if BH-adjusted p < 0.05 for SSIM OR for PSNR
      -> (2) order the survivors by ``combined``
      -> (3) greedy redundancy removal (|Pearson r| >= 0.90, keep higher rank)
      -> final selected set (size determined by the data, NOT fixed to 14)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


def permutation_ranking(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 200, min_samples_leaf: int = 2,
    n_repeats: int = 5, random_state: int = 42,
) -> pd.DataFrame:
    """Fit a Random Forest on TRAIN, score permutation importance on VALIDATION.

    Returns one row per feature: importance mean/std, its rank for this target,
    and the model's validation R^2 (context for how much the importances mean).
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    r2_val = float(rf.score(X_val, y_val))
    perm = permutation_importance(
        rf, X_val, y_val, n_repeats=n_repeats,
        random_state=random_state, n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    })
    # rank 1 = most important; ties resolved alphabetically for determinism
    df = df.sort_values(["importance_mean", "feature"],
                        ascending=[False, True]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["rf_val_r2"] = round(r2_val, 6)
    return df


def combine_rankings(df_ssim: pd.DataFrame, df_psnr: pd.DataFrame) -> pd.DataFrame:
    """Average the two 1-based ranks; ties broken by mean normalised importance.

    See the module docstring for the exact formula.
    """
    a = df_ssim.set_index("feature")
    b = df_psnr.set_index("feature")
    out = pd.DataFrame(index=a.index)
    out["rank_ssim"] = a["rank"]
    out["rank_psnr"] = b["rank"]
    out["importance_ssim"] = a["importance_mean"]
    out["importance_psnr"] = b["importance_mean"]
    out["rf_val_r2_ssim"] = a["rf_val_r2"]
    out["rf_val_r2_psnr"] = b["rf_val_r2"]
    out["combined_rank"] = (out["rank_ssim"] + out["rank_psnr"]) / 2.0
    # normalised (0-1) importance inside each target, then averaged -- only
    # used as a tie-break, never as the primary ordering.
    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = float(s.min()), float(s.max())
        return (s - lo) / (hi - lo) if hi > lo else s * 0.0
    out["importance_norm_mean"] = (_norm(out["importance_ssim"])
                                   + _norm(out["importance_psnr"])) / 2.0
    out = out.reset_index().rename(columns={"index": "feature"})
    out = out.sort_values(["combined_rank", "importance_norm_mean", "feature"],
                          ascending=[True, False, True]).reset_index(drop=True)
    out["combined_rank_order"] = np.arange(1, len(out) + 1)
    return out


def select_final_set(
    ranking: pd.DataFrame,
    adjusted_p: pd.DataFrame,
    corr: pd.DataFrame,
    alpha: float = 0.05,
    threshold: float = 0.90,
) -> dict:
    """Assemble the funnel: FDR screen -> combined rank -> redundancy removal.

    ``adjusted_p`` must have one row per feature with columns
    ``adjusted_p_ssim`` and ``adjusted_p_psnr`` (already BH-corrected).
    ``corr`` is the TRAIN Pearson feature-feature matrix.
    Returns a dict with every intermediate list so the report can show the
    full funnel, not just the endpoint.
    """
    from stats.redundancy import remove_redundant

    merged = ranking.merge(adjusted_p, on="feature", how="left")
    significant = merged[
        (merged["adjusted_p_ssim"] < alpha) | (merged["adjusted_p_psnr"] < alpha)
    ].copy()
    # keep the rank order for both the significant screen and the survivors
    significant = significant.sort_values("combined_rank").reset_index(drop=True)
    ranked_all = merged.sort_values("combined_rank").reset_index(drop=True)

    kept_after_screen = significant["feature"].tolist()
    kept, removed = remove_redundant(corr, kept_after_screen, threshold)

    return {
        "all_features": ranked_all["feature"].tolist(),
        "significant": kept_after_screen,
        "final": kept,
        "removed": removed,
        "ranking": ranked_all,
    }
