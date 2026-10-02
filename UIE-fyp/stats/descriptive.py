"""Descriptive statistics for the 25 handcrafted features.

WHAT THIS MODULE IS FOR
-----------------------
Before any hypothesis test, we report what the features actually look like:
centre (mean, median), spread (std, variance, IQR), range and shape (skew,
kurtosis). This is the "25-feature table" required for the review, and it is
also a sanity check: a feature with zero variance cannot be tested at all.

It performs NO selection and NO significance testing -- it is context only.

INPUT   a pandas DataFrame with the 25 feature columns
OUTPUT  a tidy DataFrame: one row per feature, one column per statistic
"""
from __future__ import annotations

import pandas as pd


def describe_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """One row per feature: mean/median/std/variance/min/max/q1/q3/iqr/skew/kurtosis.

    ``df`` must contain every name in ``feature_names`` (NaNs are not expected;
    the upstream dataset builder already refuses to write them).
    """
    rows = []
    for name in feature_names:
        x = df[name].astype(float)
        q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
        rows.append({
            "feature": name,
            "n": int(x.notna().sum()),
            "mean": float(x.mean()),
            "median": float(x.median()),
            "std": float(x.std(ddof=1)),
            "variance": float(x.var(ddof=1)),
            "min": float(x.min()),
            "max": float(x.max()),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            # pandas' skew/kurtosis are the standard Fisher definitions
            # (kurtosis here is the EXCESS kurtosis: 0 == Gaussian).
            "skew": float(x.skew()),
            "kurtosis_excess": float(x.kurtosis()),
        })
    return pd.DataFrame(rows)
