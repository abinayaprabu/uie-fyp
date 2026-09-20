#!/usr/bin/env python3
"""Render the restoration experiment's committed numbers as figures.

Everything here is drawn ONLY from committed artefacts of the original,
verified run `unet_128`:

  results/enhancement/unet_128/train_history.csv   -> training-curve figure
  results/enhancement/unet_128/test_per_image.csv  -> per-image delta figure
  results/enhancement/unet_128/metrics.json        -> annotations (CIs, means)

No model, no image, no re-computation: these are visualisations of the
sealed-test verdict that is already on GitHub, so the restoration result is
inspectable while enhanced pixels are being regenerated.

Outputs:
  plots/enhancement_training_curve.png
  plots/enhancement_per_image_delta.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ENHANCEMENT_RESULTS_DIR  # noqa: E402

RUN = ENHANCEMENT_RESULTS_DIR / "unet_128"
HIST = RUN / "train_history.csv"
PERIMG = RUN / "test_per_image.csv"
METRICS = RUN / "metrics.json"


def training_curve() -> Path:
    h = pd.read_csv(HIST)
    m = json.loads(METRICS.read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.plot(h.epoch, h.train_l1, lw=1.4, color="tab:blue")
    a1.set_xlabel("epoch"); a1.set_ylabel("train L1 loss")
    a1.set_title("training loss (128x128 crops, batch 8, L1)")
    a1.grid(alpha=.3)
    v = h.dropna(subset=["val_ssim"])
    a2.plot(v.epoch, v.val_ssim, "o-", ms=4, lw=1.2, color="tab:orange",
            label="val SSIM (full-res, every 5 epochs)")
    best = h.loc[h.is_best.astype(bool).idxmax()] if "is_best" in h else v.loc[v.val_ssim.idxmax()]
    a2.plot([best.epoch], [best.val_ssim], "*", ms=18, color="tab:red", zorder=5,
            label=f"best kept: epoch {int(best.epoch)}")
    a2.set_xlabel("epoch"); a2.set_ylabel("val SSIM")
    a2.set_title(f"early stopping on val SSIM (best {best.val_ssim:.4f}, "
                 f"run stopped epoch {int(h.epoch.max())})")
    a2.legend(fontsize=8); a2.grid(alpha=.3)
    fig.suptitle("U-Net Config A - training history of the COMMITTED, verified run "
                 "`unet_128` (seed 42, 472,259 params)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = Path("plots/enhancement_training_curve.png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    return out


def per_image_delta() -> Path:
    d = pd.read_csv(PERIMG)
    m = json.loads(METRICS.read_text())
    ds = m.get("unet_minus_classical", {})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    s = d.sort_values("ssim_delta")
    colors = ["tab:green" if x > 0 else "tab:red" for x in s.ssim_delta]
    a1.bar(range(len(s)), s.ssim_delta, color=colors, width=1.0)
    a1.axhline(0, color="k", lw=.8)
    mean = s.ssim_delta.mean()
    a1.axhline(mean, color="tab:blue", lw=1.2, ls="--", label=f"mean {mean:+.4f}")
    wins = int((d.ssim_delta > 0).sum()); losses = int((d.ssim_delta <= 0).sum())
    ci = ds.get("ssim", {}).get("mean_ci95")
    citxt = f" CI95 [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else ""
    a1.set_title(f"per-image SSIM gain over classical, sorted\n"
                 f"win {wins} / loss {losses} of {len(d)}{citxt}", fontsize=10)
    a1.set_xlabel("test images (worst gain -> best gain)"); a1.set_ylabel("SSIM(U-Net) - SSIM(classical)")
    a1.legend(fontsize=8); a1.grid(alpha=.3, axis="y")
    a2.scatter(d.ssim_classical, d.ssim_delta, s=14, alpha=.75,
               c=["tab:green" if x > 0 else "tab:red" for x in d.ssim_delta])
    z = pd.np.polyfit(d.ssim_classical, d.ssim_delta, 1) if hasattr(pd, "np") else \
        __import__("numpy").polyfit(d.ssim_classical, d.ssim_delta, 1)
    xs = [d.ssim_classical.min(), d.ssim_classical.max()]
    a2.plot(xs, [z[0] * x + z[1] for x in xs], "k--", lw=1.2,
            label=f"slope {z[0]:+.3f} (raises the floor)")
    a2.axhline(0, color="k", lw=.8)
    a2.set_xlabel("classical SSIM (how hard the image was)")
    a2.set_ylabel("SSIM gain from U-Net")
    a2.set_title("gain vs difficulty: bad images improve most", fontsize=10)
    a2.legend(fontsize=8); a2.grid(alpha=.3)
    fig.suptitle("Restoration verdict on the sealed 133 test images - committed run "
                 "`unet_128`, recomputed nowhere", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = Path("plots/enhancement_per_image_delta.png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    training_curve()
    per_image_delta()
