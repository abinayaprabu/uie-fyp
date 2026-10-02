"""Multiple-testing correction: Benjamini-Hochberg FDR (and a Bonferroni check).

WHY THIS EXISTS
---------------
25 features are tested per target, so 25 p-values are produced. Testing many
hypotheses at once inflates the chance of "significant" results that are pure
noise. The standard remedy used in the project is the Benjamini-Hochberg
procedure at alpha = 0.05, which controls the expected proportion of FALSE
DISCOVERIES among the features declared significant.

PROCEDURE (Benjamini & Hochberg, 1995)
--------------------------------------
1. sort the m p-values ascending:  p(1) <= p(2) <= ... <= p(m)
2. adjusted p-value of the i-th (1-based) is  q(i) = min over k >= i of
   ( p(k) * m / k )  -- the running minimum enforces monotonicity
3. a feature is significant when q(i) < alpha.

Bonferroni (p * m, capped at 1) is also reported as a conservative SENSITIVITY
analysis: features that survive Bonferroni are a strict subset of the FDR
set, so reporting both makes the choice of procedure auditable.

INPUT   a list of raw p-values (one family = one target)
OUTPUT  adjusted p-values in the ORIGINAL order, plus a small summary
"""
from __future__ import annotations

import numpy as np


def benjamini_hochberg(pvals: list[float] | np.ndarray) -> np.ndarray:
    """BH-adjusted p-values, returned in the same order as the input."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")       # stable -> deterministic
    ranked = p[order]
    # q_i = p_i * m / i, then enforce monotonicity with a running minimum
    # taken from the largest p-value downwards.
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = q
    return out


def bonferroni(pvals: list[float] | np.ndarray) -> np.ndarray:
    """Bonferroni-adjusted p-values (min(1, p * m)) -- sensitivity analysis."""
    p = np.asarray(pvals, dtype=float)
    return np.clip(p * len(p), 0.0, 1.0)
