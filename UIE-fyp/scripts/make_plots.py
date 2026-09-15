#!/usr/bin/env python3
"""Generate all thesis plots (matplotlib, no seaborn dependency).

  plots/training_curve_<run>.png   (train/val loss per epoch, every cnn run)
  plots/predicted_vs_actual_ssim.png / ..._psnr.png (test scatter, all models)
  plots/model_comparison.png        (grouped R²/RMSE bars from baseline_metrics)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CNN_RESULTS_DIR, COMPARISON_RESULTS_DIR, PLOTS_DIR  # noqa: E402


def training_curves() -> None:
    for run_dir in sorted(CNN_RESULTS_DIR.iterdir()):
        hist = run_dir / "train_history.csv"
        if not hist.exists():
            continue
        h = pd.read_csv(hist)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(h["epoch"], h["train_loss"], label="train loss")
        ax.plot(h["epoch"], h["val_loss"], label="val loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE (normalised targets)")
        ax.set_title(f"Training curve — {run_dir.name}")
        ax.legend(); fig.tight_layout()
        out = PLOTS_DIR / f"training_curve_{run_dir.name}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"Saved {out}")


def pred_vs_actual() -> None:
    frames = []
    for run_dir in sorted(CNN_RESULTS_DIR.iterdir()):
        pred = run_dir / "test_predictions.csv"
        if pred.exists():
            d = pd.read_csv(pred)
            d["model"] = run_dir.name
            frames.append(d)
    rf = COMPARISON_RESULTS_DIR / "rf_baseline_test_predictions.csv"
    if rf.exists():
        d = pd.read_csv(rf)
        d["model"] = "baseline_rf"
        frames.append(d)
    if not frames:
        print("No test predictions found; skipping scatter plots.")
        return
    for target in ("ssim", "psnr"):
        fig, ax = plt.subplots(figsize=(6.5, 6))
        for d in frames:
            ax.scatter(d[f"{target}_true"], d[f"{target}_pred"], s=12, alpha=0.6,
                       label=d["model"].iloc[0])
        lo = min(d[f"{target}_true"].min() for d in frames)
        hi = max(d[f"{target}_true"].max() for d in frames)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
        ax.set_xlabel(f"{target.upper()} true"); ax.set_ylabel(f"{target.upper()} predicted")
        ax.set_title(f"Predicted vs actual {target.upper()} (test set)")
        ax.legend(fontsize=8); fig.tight_layout()
        out = PLOTS_DIR / f"predicted_vs_actual_{target}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"Saved {out}")


def model_comparison() -> None:
    path = COMPARISON_RESULTS_DIR / "baseline_metrics.csv"
    if not path.exists():
        print("No baseline_metrics.csv; skipping comparison plot.")
        return
    df = pd.read_csv(path)
    models = df["model"].tolist()
    x = range(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar([i - 0.2 for i in x], df["ssim_r2"], 0.4, label="SSIM R²")
    axes[0].bar([i + 0.2 for i in x], df["psnr_r2"], 0.4, label="PSNR R²")
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].set_ylabel("R² (test)"); axes[0].set_title("Test R² by model")
    axes[0].legend()
    axes[1].bar([i - 0.2 for i in x], df["ssim_rmse"], 0.4, label="SSIM RMSE")
    axes[1].bar([i + 0.2 for i in x], df["psnr_rmse"], 0.4, label="PSNR RMSE")
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].set_ylabel("RMSE (original units)"); axes[1].set_title("Test RMSE by model")
    axes[1].legend()
    fig.tight_layout()
    out = PLOTS_DIR / "model_comparison.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved {out}")


def main() -> int:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    training_curves()
    pred_vs_actual()
    model_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
