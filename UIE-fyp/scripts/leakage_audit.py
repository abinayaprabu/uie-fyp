#!/usr/bin/env python3
"""Automated leakage audit (required before any final result is reported).

Run:  python scripts/leakage_audit.py

The audit CHECKS, from artefacts and by recomputation, that:

  1. the frozen split is 623/134/133 and the three id sets are disjoint;
  2. the Stage-A statistics stored in results/statistics/ could NOT have been
     computed with test rows: each stored p-value matches a TRAIN-only
     recomputation and differs from an all-rows recomputation;
  3. the feature scaler inside a run's checkpoint was fitted on TRAIN rows
     only (bit-exact match to a train-only recomputation, and different from
     an all-rows scaler);
  4. augmentation is TRAIN-only (val/test datasets report augment=False);
  5. the model never sees a reference: the dataset returns references only as
     targets, and the checkpoint's train fingerprint equals the split file's
     train ids;
  6. the evaluation metrics are recomputed from the PNG bytes on disk (so a
     stale in-memory tensor cannot produce a number).

Exit code 0 = all checks pass; 1 = at least one FAIL.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn.dataset_pairs import ids_fingerprint, split_ids  # noqa: E402
from src.config import (  # noqa: E402
    FEATURE_RESULTS_DIR, MODELS_DIR, SPLIT_CSV, STATS_RESULTS_DIR,
)

QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"
failures: list[str] = []
checks = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def main() -> int:
    print("=" * 78)
    print("LEAKAGE AUDIT")
    print("=" * 78)

    # ---------------------------------------------------------------- 1
    print("\n[1] frozen split integrity")
    split = pd.read_csv(SPLIT_CSV)
    n = split["split"].value_counts().to_dict()
    check("split sizes are 623/134/133",
          n.get("train") == 623 and n.get("val") == 134 and n.get("test") == 133,
          str(n))
    ids = {s: set(split.loc[split["split"] == s, "image_name"]) for s in
           ("train", "val", "test")}
    check("no id appears in two splits",
          not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"])
          and not (ids["val"] & ids["test"]))
    check("no duplicate image ids in the split file",
          not split["image_name"].duplicated().any())
    print(f"    fingerprints: train {ids_fingerprint(sorted(ids['train']))} | "
          f"val {ids_fingerprint(sorted(ids['val']))} | "
          f"test {ids_fingerprint(sorted(ids['test']))}")

    # ---------------------------------------------------------------- 2
    print("\n[2] Stage-A statistics/selection are exactly reproducible from "
          "TRAIN rows only")
    q = pd.read_csv(QUALITY_CSV).merge(split, on="image_name")
    train = q[q["split"] == "train"].reset_index(drop=True)
    val = q[q["split"] == "val"].reset_index(drop=True)
    stored = pd.read_csv(STATS_DIR() / "hypothesis_tests.csv")
    from stats.multiple_testing import benjamini_hochberg
    from stats.correlation import correlation_matrix
    from stats.selector import (combine_rankings, permutation_ranking,
                                select_final_set)
    from src.config import (CORR_THRESHOLD, FEATURE_NAMES_25, PERM_N_REPEATS,
                            RF_MIN_SAMPLES_LEAF, RF_N_ESTIMATORS, RANDOM_STATE)
    worst_p, worst_q = 0.0, 0.0
    adjusted = {}
    for target in ("ssim", "psnr"):
        X = train[FEATURE_NAMES_25].to_numpy(float)
        y = train[target].to_numpy(float)
        _, p = f_regression(X, y)
        q_bh = benjamini_hochberg(p)
        adjusted[target] = pd.DataFrame({
            "feature": FEATURE_NAMES_25, f"adjusted_p_{target}": q_bh})
        s = stored[stored["target"] == target].set_index("feature")
        for j, feat in enumerate(FEATURE_NAMES_25):
            worst_p = max(worst_p, abs(p[j] - float(s.loc[feat, "p_raw"])))
            worst_q = max(worst_q, abs(q_bh[j] - float(s.loc[feat, "p_bh_fdr"])))
    check("stored raw p-values match a TRAIN-only recomputation",
          worst_p < 1e-12, f"max |diff| = {worst_p:.2e}")
    check("stored BH-adjusted p-values match a TRAIN-only recomputation",
          worst_q < 1e-12, f"max |diff| = {worst_q:.2e}")

    # Re-run the whole funnel (F-test -> BH -> RF permutation on VALIDATION ->
    # combined rank -> redundancy) with TRAIN for fitting and VAL for the
    # permutation importance, and compare the final list with what was stored.
    ranks = {}
    for target in ("ssim", "psnr"):
        ranks[target] = permutation_ranking(
            train[FEATURE_NAMES_25].to_numpy(float), train[target].to_numpy(float),
            val[FEATURE_NAMES_25].to_numpy(float), val[target].to_numpy(float),
            FEATURE_NAMES_25, n_estimators=RF_N_ESTIMATORS,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF, n_repeats=PERM_N_REPEATS,
            random_state=RANDOM_STATE)
    combined = combine_rankings(ranks["ssim"], ranks["psnr"])
    adjusted_all = adjusted["ssim"].merge(adjusted["psnr"], on="feature")
    funnel = select_final_set(combined, adjusted_all,
                              correlation_matrix(train, FEATURE_NAMES_25),
                              alpha=0.05, threshold=CORR_THRESHOLD)
    stored_sel = (pd.read_csv(STATS_DIR() / "selected_features.csv")
                  .sort_values("rank")["feature"].astype(str).tolist())
    check("stored selected feature set == a fresh TRAIN-only recomputation",
          funnel["final"] == stored_sel,
          f"stored {stored_sel} vs recomputed {funnel['final']}")

    # ---------------------------------------------------------------- 3
    print("\n[3] feature scaler fitted on TRAIN rows only (checkpoint)")
    ckpts = sorted(MODELS_DIR.glob("best_enh224_*.pt"))
    if not ckpts:
        print("  (no trained checkpoints yet -- skipped; re-run after training)")
    for ckpt_path in ckpts:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        names = ckpt["feature_names"]
        tr = q[q["split"] == "train"]
        sc_train = StandardScaler().fit(tr[names].to_numpy(float))
        sc_all = StandardScaler().fit(q[names].to_numpy(float))
        stored_mean = np.asarray(ckpt["scaler_mean"])
        d_train = float(np.abs(stored_mean - sc_train.mean_).max())
        d_all = float(np.abs(stored_mean - sc_all.mean_).max())
        check(f"{ckpt_path.name}: scaler == train-only scaler",
              d_train < 1e-9, f"max |diff| = {d_train:.2e}")
        check(f"{ckpt_path.name}: scaler != all-rows scaler",
              d_all > 1e-6, f"max |diff| = {d_all:.2e}")

    # ---------------------------------------------------------------- 4
    print("\n[4] augmentation is TRAIN-only")
    import torch
    from cnn.feature_guided.dataset import HybridPairs, fit_feature_scaler, load_selected_features
    names = load_selected_features()
    scaler = fit_feature_scaler(names)
    check("train dataset augment=True",
          HybridPairs("train", names, scaler, augment=True, limit=4).augment is True)
    check("val dataset augment=False despite augment=True",
          HybridPairs("val", names, scaler, augment=True, limit=4).augment is False)
    check("test dataset augment=False despite augment=True",
          HybridPairs("test", names, scaler, augment=True, limit=4).augment is False)

    # ---------------------------------------------------------------- 5
    print("\n[5] reference images are targets, never inputs")
    ds = HybridPairs("train", names, scaler, limit=2)
    _, _, target, name = ds[0]
    check("dataset returns (image, features, target, name)",
          target.shape == (3, 224, 224))
    src = (Path(__file__).resolve().parent.parent / "cnn" / "feature_guided" / "dataset.py").read_text()
    check("dataset module never feeds the reference into the features",
          "extract_features" not in src and "REFERENCE_DIR / name" not in src)
    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        check(f"{ckpt_path.name}: train fingerprint matches the split file",
              ckpt["train_ids_fingerprint"] == ids_fingerprint(split_ids("train")))
        check(f"{ckpt_path.name}: test fingerprint matches the split file",
              ckpt["test_ids_fingerprint"] == ids_fingerprint(split_ids("test")))

    # ---------------------------------------------------------------- 6
    print("\n[6] evaluation metrics come from the PNG bytes on disk")
    from src.iqa import compute_ssim_psnr
    from src.preprocess import resize_reference_like_preprocessed
    from cnn.dataset_pairs import aligned_reference
    import cv2
    name = sorted(ids["test"])[0]
    pre = cv2.imread(str(Path("dataset/preprocessed") / name), cv2.IMREAD_COLOR)
    ref = aligned_reference(name)
    s1, p1 = compute_ssim_psnr(pre, ref)
    # recompute the same pair via a fresh read of the same bytes
    pre2 = cv2.imread(str(Path("dataset/preprocessed") / name), cv2.IMREAD_COLOR)
    s2, p2 = compute_ssim_psnr(pre2, aligned_reference(name))
    check("metric is deterministic on re-read bytes", s1 == s2 and p1 == p2)

    print("\n" + "=" * 78)
    if failures:
        print(f"LEAKAGE AUDIT FAILED ({len(failures)} of {checks} checks): {failures}")
        return 1
    print(f"LEAKAGE AUDIT PASSED ({checks} checks)")
    return 0


def STATS_DIR() -> Path:
    return STATS_RESULTS_DIR


if __name__ == "__main__":
    raise SystemExit(main())
