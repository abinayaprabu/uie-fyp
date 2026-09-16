#!/usr/bin/env python3
"""Create the single fixed train/validation/test split (by image identity).

- Uses the validated paired identities from dataset_validation_report.csv
  (falls back to raw-890 filenames if the report is missing).
- 70% / 15% / 15% with RANDOM_STATE=42  ->  train 623 / val 134 / test 133.
  (An earlier docstring listed these as "623 / 133 / 134", transposing val
  and test; the counts here are verified against data_split.csv.)
- Saved to results/feature/data_split.csv; EVERY later stage (ranking,
  correlation, subset evaluation, CNN, baselines) must read this file so the
  test identities stay identical and untouched until final evaluation.

CHECK 11 (no id overlap between splits) is asserted here.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    RANDOM_STATE,
    RAW_DIR,
    SPLIT_CSV,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
    VALIDATION_REPORT_CSV,
)


def duplicate_groups(ids: list[str]) -> list[list[str]]:
    """Group image names by raw-file SHA-256 (singletons stay alone)."""
    by_hash: dict[str, list[str]] = {}
    for name in ids:
        h = hashlib.sha256((RAW_DIR / name).read_bytes()).hexdigest()
        by_hash.setdefault(h, []).append(name)
    return [sorted(v) for v in by_hash.values()]


def main() -> int:
    if VALIDATION_REPORT_CSV.exists():
        report = pd.read_csv(VALIDATION_REPORT_CSV)
        ids = report.loc[report["raw_exists"] & report["reference_exists"],
                         "image_name"].tolist()
        print(f"Loaded {len(ids)} validated paired identities from validation report.")
    else:
        ids = sorted(p.name for p in RAW_DIR.glob("*") if p.is_file())
        print(f"WARNING: no validation report; using {len(ids)} raw filenames.")

    rng = np.random.RandomState(RANDOM_STATE)
    groups = duplicate_groups(sorted(ids))
    order = rng.permutation(len(groups))
    n = len(ids)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    # Greedy fill by groups so duplicate members never straddle splits.
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    test_ids: set[str] = set()
    for g in [groups[i] for i in order]:
        if len(train_ids) + len(g) <= n_train:
            train_ids.update(g)
        elif len(val_ids) + len(g) <= n_val:
            val_ids.update(g)
        else:
            test_ids.update(g)
    multi = [g for g in groups if len(g) > 1]
    print(f"Duplicate groups kept whole: {len(multi)} groups "
          f"({sum(len(g) for g in multi)} images)")

    # CHECK 11: splits must be disjoint and cover everything.
    assert not (train_ids & val_ids), "train/val overlap!"
    assert not (train_ids & test_ids), "train/test overlap!"
    assert not (val_ids & test_ids), "val/test overlap!"
    assert len(train_ids | val_ids | test_ids) == n, "split does not cover all ids!"

    rows = ([(i, "train") for i in sorted(train_ids)] +
            [(i, "val") for i in sorted(val_ids)] +
            [(i, "test") for i in sorted(test_ids)])
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["image_name", "split"]).to_csv(SPLIT_CSV, index=False)
    print(f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} (total {n})")
    print(f"Saved {SPLIT_CSV}")
    print("CHECK 11 (disjoint train/val/test identities): PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
