"""Redundancy removal: drop features that duplicate each other (|Pearson r| >= 0.90).

WHY
---
Relevance screening can leave several features that carry the SAME
information (e.g. ``variance`` and ``std`` are algebraically linked:
variance = std^2, so r = 1.000; ``ASM`` and ``energy`` likewise:
ASM = energy^2; ``rms_contrast`` is std divided by the mean). Feeding all of
them to a model adds no information and makes any importance ranking unstable,
because the credit for one effect is split arbitrarily across duplicates.

RULE (stated once, applied mechanically)
----------------------------------------
Correlation is computed on TRAIN rows only. Features are visited in RANK
ORDER (most important first, as produced by ``selector.py``). A feature is
REMOVED if it has |Pearson r| >= 0.90 with any feature already KEPT. The kept
feature always comes earlier in the ranking, i.e. "keep the higher-ranked
feature" -- exactly the documented rule.

The full 25-feature analysis is written out before this step runs, so nothing
is removed silently.

INPUT   correlation matrix (DataFrame), ranked feature order, threshold
OUTPUT  (kept list, removed list of dicts explaining each removal)
"""
from __future__ import annotations

import pandas as pd


def remove_redundant(corr: pd.DataFrame, ranked_features: list[str],
                     threshold: float = 0.90) -> tuple[list[str], list[dict]]:
    """Greedy redundancy filter in rank order.

    ``corr`` must be a square labelled Pearson matrix computed on TRAIN rows.
    Returns ``(kept, removed)``; each removal records the surviving feature,
    the correlation value, and why it was removed.
    """
    kept: list[str] = []
    removed: list[dict] = []
    for name in ranked_features:
        clash = None
        for survivor in kept:
            r = float(corr.loc[name, survivor])
            if abs(r) >= threshold:
                clash = (survivor, r)
                break
        if clash is None:
            kept.append(name)
        else:
            survivor, r = clash
            removed.append({
                "removed_feature": name,
                "kept_feature": survivor,
                "pearson_r": round(r, 6),
                "abs_r": round(abs(r), 6),
                "reason": (f"|r| = {abs(r):.3f} >= {threshold} with the "
                           f"higher-ranked feature '{survivor}'"),
            })
    return kept, removed
