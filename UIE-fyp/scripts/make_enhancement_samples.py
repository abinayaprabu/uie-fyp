#!/usr/bin/env python3
"""Build the tracked enhancement sample figure: raw | classical | U-Net | reference.

The 133 enhanced test PNGs (dataset/enhanced-test/, ~40 MB) and the 890
preprocessed PNGs are gitignored binaries; per the .gitignore convention,
*sample side-by-side figures* ARE tracked under plots/. This script renders
that figure for the restoration result so the U-Net's output is visible
without shipping every enhanced image.

Row selection uses ONLY the frozen label CSV (feature_quality_dataset.csv,
md5 e90a073f...), ordered by classical SSIM vs the pseudo-reference - worst /
lower-middle / upper-middle / best. No enhancement metric participates in the
selection, so the sealed test set is not re-read for any decision; the figure
is presentation of already-committed artefacts plus regenerated images.

Requires dataset/enhanced-test/ to exist first:
    python -m cnn.enhance --split test

Usage:
    python scripts/make_enhancement_samples.py [--rows 4] [--out plots/enhancement_samples.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pd_plot  # noqa: E402  (matplotlib.pyplot)
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    ENHANCED_DIR,
    FEATURE_RESULTS_DIR,
    PREPROCESSED_DIR,
    RAW_DIR,
    REFERENCE_DIR,
)

QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=4, help="number of sample rows (default 4)")
    ap.add_argument("--out", type=Path, default=Path("plots/enhancement_samples.png"))
    args = ap.parse_args()

    if not ENHANCED_DIR.exists():
        print(f"ERROR: {ENHANCED_DIR} missing. Run: python -m cnn.enhance --split test",
              file=sys.stderr)
        return 1

    df = pd.read_csv(QUALITY_CSV).sort_values("ssim").reset_index(drop=True)
    n = len(df)
    pos = [round(i * (n - 1) / (args.rows - 1)) for i in range(args.rows)]

    cols = [
        (RAW_DIR, "RAW (underwater)"),
        (PREPROCESSED_DIR, "classical pipeline"),
        (ENHANCED_DIR, "U-Net (Config A)"),
        (REFERENCE_DIR, "pseudo-reference\n(target only, never an input)"),
    ]
    fig, axes = pd_plot.subplots(args.rows, 4, figsize=(15, 4.0 * args.rows))
    for j, (_, title) in enumerate(cols):
        axes[0, j].set_title(title, fontsize=11)
    for i, p in enumerate(pos):
        row = df.iloc[p]
        name = row["image_name"]
        missing = [str(d) for d, _ in cols if not (d / name).exists()]
        if missing:
            print(f"ERROR: missing inputs for {name}: {missing}", file=sys.stderr)
            return 1
        for j, (d, _) in enumerate(cols):
            img = cv2.cvtColor(cv2.imread(str(d / name)), cv2.COLOR_BGR2RGB)
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(
            f"{name}\nclassical SSIM {row['ssim']:.3f}\nclassical PSNR {row['psnr']:.2f} dB",
            rotation=0, ha="right", va="center", fontsize=10, labelpad=34)
    fig.suptitle(
        "Restoration sample: classical preprocessing vs learned enhancement (U-Net, Config A)\n"
        "rows ordered by classical SSIM (worst -> best); U-Net column from dataset/enhanced-test/",
        fontsize=12)
    fig.tight_layout(rect=[0.07, 0, 1, 0.95])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out} ({args.out.stat().st_size/1e6:.2f} MB), rows={pos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
