#!/usr/bin/env python3
"""Build feature_dataset.csv + feature_quality_dataset.csv.

For every paired image identity:
  1. Load PREPROCESSED image, convert BGR->RGB, extract the 25 handcrafted
     features (src.features.extract_features).
  2. Load REFERENCE image, apply the identical geometric resize (width 600,
     INTER_AREA), compute SSIM and PSNR of (preprocessed, reference) with
     skimage (channel_axis=2, data_range=255).

Outputs:
  - feature_dataset.csv                       (image_name + 25 features)
  - results/feature/feature_quality_dataset.csv (image_name + 25 + ssim + psnr)

Performs CHECKS 3-10 (row count, 25 columns, no missing values, no duplicate
ids, feature ranges, SSIM in ~[0,1], finite PSNR, no NaN/inf) and exits 1 on
failure. This stage uses NO train/test split information — splitting happens
downstream and every later stage reads results/feature/data_split.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    FEATURE_DATASET_CSV,
    FEATURE_NAMES_25,
    FEATURE_RESULTS_DIR,
    PREPROCESSED_DIR,
    RAW_DIR,
    REFERENCE_DIR,
)
from src.features import extract_features  # noqa: E402
from src.preprocess import resize_reference_like_preprocessed  # noqa: E402


def compute_ssim_psnr(pre_bgr: np.ndarray, ref_bgr: np.ndarray) -> tuple[float, float]:
    """SSIM/PSNR between preprocessed and reference (both BGR uint8).

    Images are converted to RGB only for convention consistency (SSIM/PSNR are
    channel-order invariant as long as both inputs share the order). Shapes
    must already match; the caller guarantees this via the shared resize.
    """
    if pre_bgr.shape != ref_bgr.shape:
        raise ValueError(f"shape mismatch: preprocessed {pre_bgr.shape} vs "
                         f"reference {ref_bgr.shape}")
    pre = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2RGB)
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    ssim = float(structural_similarity(pre, ref, channel_axis=2, data_range=255))
    psnr = float(peak_signal_noise_ratio(ref, pre, data_range=255))
    return ssim, psnr


def main() -> int:
    FEATURE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pre_files = {p.name: p for p in PREPROCESSED_DIR.glob("*") if p.is_file()}
    ref_files = {p.name: p for p in REFERENCE_DIR.glob("*") if p.is_file()}
    ids = sorted(set(pre_files) & set(ref_files))
    if not ids:
        print("ERROR: no paired preprocessed/reference images. Run "
              "scripts/run_preprocessing.py first.", file=sys.stderr)
        return 1
    print(f"Building feature dataset for {len(ids)} paired images ...")
    t0 = time.time()

    rows = []
    for name in tqdm(ids, desc="features+targets", unit="img"):
        pre = cv2.imread(str(pre_files[name]), cv2.IMREAD_COLOR)
        ref = cv2.imread(str(ref_files[name]), cv2.IMREAD_COLOR)
        if pre is None or ref is None:
            print(f"ERROR: unreadable pair {name}", file=sys.stderr)
            return 1
        ref_r = resize_reference_like_preprocessed(ref)
        feats = extract_features(cv2.cvtColor(pre, cv2.COLOR_BGR2RGB))
        ssim, psnr = compute_ssim_psnr(pre, ref_r)
        rows.append({"image_name": name, **feats, "ssim": ssim, "psnr": psnr})

    df = pd.DataFrame(rows)
    feat_cols = ["image_name"] + FEATURE_NAMES_25
    df[feat_cols].to_csv(FEATURE_DATASET_CSV, index=False)
    df.to_csv(FEATURE_RESULTS_DIR / "feature_quality_dataset.csv", index=False)
    print(f"Saved {FEATURE_DATASET_CSV} and feature_quality_dataset.csv "
          f"in {time.time()-t0:.0f}s")

    # ---------------- CHECKS 3-10 ----------------
    failures: list[str] = []
    n_raw = len([p for p in RAW_DIR.glob('*') if p.is_file()])
    print(f"CHECK 3 (row count {len(df)} vs raw {n_raw}):",
          "PASS" if len(df) == n_raw else "FAIL")
    if len(df) != n_raw:
        failures.append("row count != raw count")

    missing = [c for c in FEATURE_NAMES_25 if c not in df.columns]
    print(f"CHECK 4 (exactly 25 feature columns): {'PASS' if not missing else 'FAIL ' + str(missing)}")
    failures.extend(f"missing column {c}" for c in missing)

    n_missing = int(df[FEATURE_NAMES_25].isna().sum().sum())
    print(f"CHECK 5 (missing feature values: {n_missing}):", "PASS" if n_missing == 0 else "FAIL")
    if n_missing:
        failures.append(f"{n_missing} missing feature values")

    n_dup = int(df["image_name"].duplicated().sum())
    print(f"CHECK 6 (duplicate image_name: {n_dup}):", "PASS" if n_dup == 0 else "FAIL")
    if n_dup:
        failures.append(f"{n_dup} duplicate image_name values")

    print("CHECK 7 (feature ranges):")
    for c in FEATURE_NAMES_25:
        print(f"  {c:20s} min={df[c].min():12.4f} max={df[c].max():12.4f}")

    smin, smax = float(df["ssim"].min()), float(df["ssim"].max())
    print(f"CHECK 8 (SSIM range [{smin:.4f}, {smax:.4f}] within ~[0,1]):",
          "PASS" if smin >= -0.05 and smax <= 1.0 else "FAIL")
    if not (smin >= -0.05 and smax <= 1.0):
        failures.append(f"SSIM out of range [{smin}, {smax}]")

    psnr_bad = int((~np.isfinite(df["psnr"])).sum())
    print(f"CHECK 9 (PSNR finite; range [{df['psnr'].min():.2f}, {df['psnr'].max():.2f}] dB; "
          f"non-finite={psnr_bad}):", "PASS" if psnr_bad == 0 else "FAIL")
    if psnr_bad:
        failures.append(f"{psnr_bad} non-finite PSNR values")

    bad_cells = int((~np.isfinite(df[FEATURE_NAMES_25 + ["ssim", "psnr"]].to_numpy(dtype=float))).sum())
    print(f"CHECK 10 (NaN/inf cells: {bad_cells}):", "PASS" if bad_cells == 0 else "FAIL")
    if bad_cells:
        failures.append(f"{bad_cells} NaN/inf cells")

    if failures:
        print("FEATURE DATASET CHECKS FAILED:", failures, file=sys.stderr)
        return 1
    print("ALL FEATURE DATASET CHECKS (3-10) PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
