"""Association measures: Pearson r and Spearman rho, plus the feature-feature matrix.

WHY TWO COEFFICIENTS
--------------------
* **Pearson r** measures the strength of the LINEAR association between a
  feature and a target (and between two features). It is reported as the
  EFFECT SIZE that accompanies the regression F-test -- same linear model,
  expressed on the familiar [-1, 1] scale.
* **Spearman rho** measures the strength of the MONOTONIC association (rank
  based). It is a ROBUSTNESS CHECK: if a feature-target relationship is
  curving or driven by outliers, Spearman will disagree with Pearson, and the
  report says so instead of hiding it.

The feature-feature Pearson matrix (TRAIN rows only) is what
``redundancy.py`` consumes to enforce the |r| >= 0.90 rule.

INPUT   X (n, 25) train feature matrix, y (n,) target, names
OUTPUT  per-feature dicts and the 25x25 matrix
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pearson_spearman(X: np.ndarray, y: np.ndarray,
                     feature_names: list[str]) -> list[dict]:
    """Pearson r and Spearman rho of every feature against one target."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    out = []
    for j, name in enumerate(feature_names):
        r = stats.pearsonr(X[:, j], y)
        rho = stats.spearmanr(X[:, j], y)
        out.append({
            "feature": name,
            "pearson_r": float(r.statistic),
            "pearson_p": float(r.pvalue),
            "spearman_rho": float(rho.statistic),
            "spearman_p": float(rho.pvalue),
        })
    return out


def correlation_matrix(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Pearson feature-feature correlation matrix (square, labelled)."""
    return df[feature_names].astype(float).corr(method="pearson")


def spearman_matrix(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Spearman feature-feature correlation matrix (robustness view)."""
    return df[feature_names].astype(float).corr(method="spearman")
