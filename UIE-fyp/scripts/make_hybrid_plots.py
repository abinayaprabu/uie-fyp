#!/usr/bin/env python3
"""Plots for the final feature-guided training runs.

Reads the training histories written by ``cnn/feature_guided/train.py`` and
draws, side by side, the image-only twin and the feature-guided model:

    results/visualizations/training_curves.png    train loss per epoch
    results/visualizations/validation_curves.png  validation SSIM and PSNR

Nothing is recomputed: the CSV histories are the source of truth, and a run
that does not exist yet is simply skipped (so the script can be run while the
second twin is still training).

Usage:  python scripts/make_hybrid_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    ENHANCEMENT_RESULTS_DIR, HYBRID_RUN_GUIDED, HYBRID_RUN_IMAGE_ONLY, RESULTS_DIR,
)

OUT_DIR = RESULTS_DIR / "visualizations"
RUNS = [(HYBRID_RUN_IMAGE_ONLY, "image-only (ablation)", "tab:blue"),
        (HYBRID_RUN_GUIDED, "feature-guided (proposed)", "tab:red")]


def load(tag: str) -> pd.DataFrame | None:
    p = ENHANCEMENT_RESULTS_DIR / tag / "train_history.csv"
    if not p.exists():
        print(f"[skip] {tag}: no history yet ({p})")
        return None
    df = pd.read_csv(p)
    return df if len(df) else None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    histories = {tag: load(tag) for tag, _, _ in RUNS}
    present = {t: h for t, h in histories.items() if h is not None}
    if not present:
        print("no histories found -- run the training first")
        return 1

    # ---- training loss ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for tag, label, colour in RUNS:
        h = histories[tag]
        if h is None:
            continue
        ax.plot(h["epoch"], h["train_loss"], color=colour, label=label)
        if "is_best" in h:
            b = h[h["is_best"].astype(str).str.lower() == "true"]
            if len(b):
                ax.scatter(b["epoch"], b["train_loss"], color=colour, zorder=3,
                           marker="*", s=90, label=f"{label}: best val SSIM")
    ax.set_xlabel("epoch"); ax.set_ylabel("training L1 loss")
    ax.set_title("Training loss (identical settings; only the feature branch differs)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_DIR/'training_curves.png'}")

    # ---- validation SSIM / PSNR -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for tag, label, colour in RUNS:
        h = histories[tag]
        if h is None:
            continue
        v = h.dropna(subset=["val_ssim"])
        if len(v):
            axes[0].plot(v["epoch"], v["val_ssim"], "o-", color=colour, label=label)
        v = h.dropna(subset=["val_psnr"])
        if len(v):
            axes[1].plot(v["epoch"], v["val_psnr"], "o-", color=colour, label=label)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("validation SSIM")
    axes[0].set_title("Validation SSIM (model selection metric)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation PSNR (dB)")
    axes[1].set_title("Validation PSNR (full resolution)")
    for a in axes:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR / "validation_curves.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_DIR/'validation_curves.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
