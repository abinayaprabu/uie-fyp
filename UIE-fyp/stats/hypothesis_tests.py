"""Univariate regression F-test — the PRIMARY relevance screen.

WHY AN F-TEST AND NOT SOMETHING ELSE
------------------------------------
The outcomes here (SSIM, PSNR) are CONTINUOUS. That rules out tests designed
for categorical outcomes (chi-square) and tests that compare group means
(t-test / ANOVA): there are no groups. The correct question is

    H0: in a simple linear regression  y = b0 + b1 * x + e  with x = one feature
        and y = one quality target,  the slope is zero  (b1 = 0)
    H1: b1 != 0.

``sklearn.feature_selection.f_regression`` answers exactly that, per feature:
it returns the F statistic of that one-variable regression and its p-value.

HONESTY NOTES (they matter in the viva)
---------------------------------------
* This tests a UNIVARIATE LINEAR association. It does NOT prove that a feature
  is important inside a neural network, and it does not capture
  non-linear/interaction effects. A separate method (Random-Forest permutation
  importance, see ``selector.py``) is used for predictive importance.
* ``f_regression`` computes F from the Pearson correlation between the feature
  and the target, so the F-test p-value and the Pearson p-value are TWO VIEWS
  OF THE SAME univariate linear test -- they are not independent evidence, and
  the report must never treat them as if they were.
* The p-value is computed for EVERY feature, so 25 tests are performed per
  target. That is why ``multiple_testing.py`` applies Benjamini-Hochberg FDR
  control before anything is called "significant".

INPUT   X: (n, 25) feature matrix (TRAIN rows), y: (n,) target
OUTPUT  list of dicts: feature, F, p_raw (+ r, kept here for convenience)
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.feature_selection import f_regression


def univariate_f_tests(X: np.ndarray, y: np.ndarray,
                       feature_names: list[str]) -> list[dict]:
    """One regression F-test per feature against one target.

    ``X`` rows must all be finite (the caller filters), ``y`` is the target
    column (SSIM or PSNR). Returns one dict per feature with the F statistic,
    the raw p-value, and the Pearson r (the effect size of the same test).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError(f"X shape {X.shape} does not match "
                         f"{len(feature_names)} feature names")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("non-finite values in X or y")

    F, p = f_regression(X, y)            # F-test per column, df1=1, df2=n-2
    # Pearson r is shown alongside because it is the EFFECT SIZE of the same
    # linear relationship: r^2 * (n - 2) / (1 - r^2) == F. Kept as a separate
    # column because "the association is strong" (large |r|) and "it is
    # statistically distinguishable from zero" (small p) are different claims.
    r = np.array([stats.pearsonr(X[:, j], y).statistic
                  for j in range(X.shape[1])])
    return [
        {"feature": name,
         "F": float(F[j]),
         "p_raw": float(p[j]),
         "pearson_r": float(r[j])}
        for j, name in enumerate(feature_names)
    ]
