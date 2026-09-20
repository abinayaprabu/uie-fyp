#!/usr/bin/env python3
"""Evaluate the enhancement U-Net on the SEALED TEST SPLIT — read once.

Scores three systems against the SAME aligned UIEB reference, with the SAME
metric definition, on the SAME 133 test images:

    1. raw            the untouched input, geometrically resized only
    2. classical      the frozen preprocessing pipeline  (the existing baseline)
    3. unet           the trained enhancement U-Net      (the new component)

This is the comparison the project objective needs: does learned enhancement
improve on classical enhancement? The classical numbers here MUST reproduce the
committed ``feature_quality_dataset.csv`` values exactly — that equality is
itself a check that this script uses the project's metric definition and not a
look-alike.

Everything is read back FROM DISK: the enhanced PNGs written by
``cnn/enhance.py``, the preprocessed PNGs, the reference PNGs. No in-memory
tensor from training is reused, so a stale or wrong checkpoint cannot silently
produce a good-looking number.

Outputs -> results/enhancement/<run_tag>/
    test_per_image.csv   one row per test image: ssim/psnr for all three systems
    metrics.json         aggregates + the paired U-Net-vs-classical test

Usage:
    python scripts/evaluate_enhancement.py [--run unet_128]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn.dataset_pairs import aligned_reference, split_ids  # noqa: E402
from src.config import (  # noqa: E402
    ENH_RUN_TAG, ENHANCED_DIR, ENHANCEMENT_RESULTS_DIR, FEATURE_RESULTS_DIR,
    MODELS_DIR, PREPROCESSED_DIR, RAW_DIR,
)
from src.iqa import compute_ssim_psnr  # noqa: E402
from src.metrics import (  # noqa: E402
    bootstrap_mean_ci, bootstrap_mean_delta_ci, paired_wilcoxon,
)
from src.preprocess import resize_to_width  # noqa: E402

QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"


def agg(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    lo, hi = bootstrap_mean_ci(a)
    return {"mean": round(float(a.mean()), 6), "median": round(float(np.median(a)), 6),
            "std": round(float(a.std(ddof=1)), 6), "min": round(float(a.min()), 6),
            "max": round(float(a.max()), 6), "n": int(len(a)),
            # 95% percentile bootstrap CI on the MEAN over the 133 test images.
            "mean_ci95": [round(lo, 6), round(hi, 6)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=ENH_RUN_TAG)
    args = ap.parse_args()

    out_dir = ENHANCEMENT_RESULTS_DIR / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    test_ids = split_ids("test")
    assert len(test_ids) == 133, f"expected 133 test images, got {len(test_ids)}"

    missing = [n for n in test_ids if not (ENHANCED_DIR / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} enhanced test images missing from {ENHANCED_DIR}, "
            f"e.g. {missing[:3]}. Run: python -m cnn.enhance --split test")

    # The committed labels for the classical baseline, for the equality check.
    committed = pd.read_csv(QUALITY_CSV).set_index("image_name")

    print(f"Evaluating {len(test_ids)} sealed test images "
          f"(raw vs classical vs {args.run}) ...")
    t0 = time.time()
    rows = []
    max_dev_ssim = max_dev_psnr = 0.0
    for i, name in enumerate(test_ids, 1):
        ref = aligned_reference(name)                       # BGR uint8, aligned
        enh = cv2.imread(str(ENHANCED_DIR / name), cv2.IMREAD_COLOR)
        pre = cv2.imread(str(PREPROCESSED_DIR / name), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(RAW_DIR / name), cv2.IMREAD_COLOR)
        if enh is None or pre is None or raw is None:
            raise ValueError(f"unreadable image for {name}")
        raw = resize_to_width(raw)                          # same frozen resize
        for tag, img in (("enh", enh), ("pre", pre), ("raw", raw)):
            if img.shape != ref.shape:
                raise ValueError(f"{name}/{tag}: shape {img.shape} != ref {ref.shape}")

        s_enh, p_enh = compute_ssim_psnr(enh, ref)
        s_pre, p_pre = compute_ssim_psnr(pre, ref)
        s_raw, p_raw = compute_ssim_psnr(raw, ref)

        # Equality check against the committed classical labels.
        d_s = abs(s_pre - float(committed.loc[name, "ssim"]))
        d_p = abs(p_pre - float(committed.loc[name, "psnr"]))
        max_dev_ssim = max(max_dev_ssim, d_s)
        max_dev_psnr = max(max_dev_psnr, d_p)

        rows.append({"image_name": name,
                     "ssim_raw": s_raw, "ssim_classical": s_pre, "ssim_unet": s_enh,
                     "psnr_raw": p_raw, "psnr_classical": p_pre, "psnr_unet": p_enh,
                     "ssim_delta": s_enh - s_pre, "psnr_delta": p_enh - p_pre,
                     "height": ref.shape[0], "width": ref.shape[1]})
        if i % 25 == 0 or i == len(test_ids):
            print(f"  {i}/{len(test_ids)} ({(time.time()-t0)/i:.2f} s/image)",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_per_image.csv", index=False)

    print(f"\n  classical-vs-committed max deviation: SSIM {max_dev_ssim:.2e}, "
          f"PSNR {max_dev_psnr:.2e}")
    metric_def_matches = max_dev_ssim < 1e-9 and max_dev_psnr < 1e-6
    print(f"  -> metric definition reproduces feature_quality_dataset.csv: "
          f"{'YES' if metric_def_matches else 'NO — INVESTIGATE'}")

    # ---- paired comparison: U-Net vs classical, on the SAME 133 images ----
    d_s = df["ssim_delta"].to_numpy()
    d_p = df["psnr_delta"].to_numpy()
    # Wilcoxon on per-image LOSSES (lower is better): 1-SSIM and -PSNR.
    p_ssim = paired_wilcoxon(1.0 - df["ssim_unet"].to_numpy(),
                             1.0 - df["ssim_classical"].to_numpy())
    p_psnr = paired_wilcoxon(-df["psnr_unet"].to_numpy(),
                             -df["psnr_classical"].to_numpy())

    metrics = {
        "run": args.run,
        "split": "test",
        "n_test": int(len(df)),
        "metric_definition_matches_committed_labels": bool(metric_def_matches),
        "max_deviation_vs_committed": {"ssim": float(max_dev_ssim),
                                       "psnr": float(max_dev_psnr)},
        "raw": {"ssim": agg(df["ssim_raw"]), "psnr": agg(df["psnr_raw"])},
        "classical": {"ssim": agg(df["ssim_classical"]),
                      "psnr": agg(df["psnr_classical"])},
        "unet": {"ssim": agg(df["ssim_unet"]), "psnr": agg(df["psnr_unet"])},
        "unet_minus_classical": {
            "ssim_mean_delta": round(float(d_s.mean()), 6),
            "ssim_median_delta": round(float(np.median(d_s)), 6),
            "ssim_delta_std": round(float(d_s.std(ddof=1)), 6),
            "ssim_win": int((d_s > 1e-9).sum()),
            "ssim_loss": int((d_s < -1e-9).sum()),
            "ssim_tie": int((np.abs(d_s) <= 1e-9).sum()),
            "ssim_wilcoxon_p": round(float(p_ssim), 6),
            "ssim_delta_ci95": [round(x, 6) for x in bootstrap_mean_delta_ci(d_s)],
            "psnr_mean_delta_db": round(float(d_p.mean()), 4),
            "psnr_median_delta_db": round(float(np.median(d_p)), 4),
            "psnr_delta_std_db": round(float(d_p.std(ddof=1)), 4),
            "psnr_win": int((d_p > 1e-9).sum()),
            "psnr_loss": int((d_p < -1e-9).sum()),
            "psnr_tie": int((np.abs(d_p) <= 1e-9).sum()),
            "psnr_wilcoxon_p": round(float(p_psnr), 6),
            "psnr_delta_ci95": [round(x, 4) for x in bootstrap_mean_delta_ci(d_p)],
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 74)
    print(f"SEALED TEST SET (n={len(df)}) — mean SSIM / mean PSNR")
    print("=" * 74)
    print(f"{'system':<22}{'SSIM mean':>11}{'SSIM med':>10}{'PSNR mean':>11}"
          f"{'PSNR med':>10}")
    for k, label in (("raw", "raw (resized only)"),
                     ("classical", "classical pipeline"),
                     ("unet", f"U-Net ({args.run})")):
        print(f"{label:<22}{metrics[k]['ssim']['mean']:>11.4f}"
              f"{metrics[k]['ssim']['median']:>10.4f}"
              f"{metrics[k]['psnr']['mean']:>11.3f}{metrics[k]['psnr']['median']:>10.3f}")
    u = metrics["unet_minus_classical"]
    print(f"\nU-Net minus classical: SSIM {u['ssim_mean_delta']:+.4f} "
          f"(win {u['ssim_win']} / loss {u['ssim_loss']} / tie {u['ssim_tie']}, "
          f"Wilcoxon p={u['ssim_wilcoxon_p']:.4f})")
    print(f"                       PSNR {u['psnr_mean_delta_db']:+.3f} dB "
          f"(win {u['psnr_win']} / loss {u['psnr_loss']} / tie {u['psnr_tie']}, "
          f"Wilcoxon p={u['psnr_wilcoxon_p']:.4f})")
    print(f"\nPaired bootstrap 95% CI on the mean difference (n=133, 4000 resamples):")
    print(f"  SSIM  {u['ssim_delta_ci95'][0]:+.4f} .. {u['ssim_delta_ci95'][1]:+.4f}"
          f"   -> {'EXCLUDES zero' if u['ssim_delta_ci95'][0] > 0 else 'includes zero'}")
    print(f"  PSNR  {u['psnr_delta_ci95'][0]:+.3f} .. {u['psnr_delta_ci95'][1]:+.3f} dB"
          f"   -> {'EXCLUDES zero' if u['psnr_delta_ci95'][0] > 0 else 'includes zero'}")
    print(f"\nSaved {out_dir/'test_per_image.csv'}")
    print(f"Saved {out_dir/'metrics.json'}")
    print("\nNEXT: python scripts/verify_enhancement.py   (independent recheck)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
