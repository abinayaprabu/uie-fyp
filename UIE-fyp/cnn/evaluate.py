"""Final evaluation of a trained checkpoint on the UNTOUCHED TEST split.

- Loads best_<run>.pt + its train-fitted scalers.
- Predicts the test split, inverse-transforms targets to ORIGINAL units
  (SSIM in [0,1]-ish, PSNR in dB), then reports R²/RMSE/MAE/Pearson r per
  target plus the project-defined avg R².
- Writes test_predictions.csv (image_name, ssim_true/pred, psnr_true/pred)
  and metrics.json into results/cnn/<run_tag>/.

Usage:  python -m cnn.evaluate --run hybrid_final
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn.dataset import UIEBQualityDataset  # noqa: E402
from cnn.model import FeatMLP, HybridCNN, ImageOnlyCNN  # noqa: E402
from src.config import (  # noqa: E402
    CNN_BATCH_SIZE, CNN_N_BOOTSTRAP, CNN_RESULTS_DIR, MODELS_DIR,
)
from src.metrics import (  # noqa: E402
    bootstrap_r2_ci, combined_r2, regression_metrics,
)

_BUILDERS = {"hybrid": HybridCNN, "image_only": ImageOnlyCNN, "mlp": FeatMLP}


def evaluate_run(run_tag: str) -> dict:
    ckpt = torch.load(MODELS_DIR / f"best_{run_tag}.pt", map_location="cpu")
    model_name, feat_names = ckpt["model_name"], ckpt["feature_names"]
    if model_name == "image_only":
        model = ImageOnlyCNN()
    elif model_name == "hybrid":
        model = HybridCNN(len(feat_names))
    elif model_name == "mlp":
        model = FeatMLP(len(feat_names))
    else:
        raise ValueError(f"unknown model in checkpoint: {model_name}")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    feat_scaler = joblib.load(MODELS_DIR / f"feat_scaler_{run_tag}.joblib")
    target_scaler = joblib.load(MODELS_DIR / f"target_scaler_{run_tag}.joblib")

    dummy = feat_names or ["mean"]
    loader = DataLoader(UIEBQualityDataset("test", dummy, feat_scaler, target_scaler),
                        batch_size=CNN_BATCH_SIZE, shuffle=False)
    names, y_true_n, y_pred_n = [], [], []
    with torch.no_grad():
        for images, feats, targets, batch_names in loader:
            y_true_n.append(targets.numpy())
            y_pred_n.append(model(images, feats).numpy())
            names.extend(list(batch_names))
    y_true = target_scaler.inverse_transform(np.vstack(y_true_n))
    y_pred = target_scaler.inverse_transform(np.vstack(y_pred_n))

    out_dir = CNN_RESULTS_DIR / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_name": names, "ssim_true": y_true[:, 0],
                  "ssim_pred": y_pred[:, 0], "psnr_true": y_true[:, 1],
                  "psnr_pred": y_pred[:, 1]}
                 ).to_csv(out_dir / "test_predictions.csv", index=False)
    metrics = {
        "run": run_tag, "model": model_name, "n_features": len(feat_names),
        "features": feat_names, "n_test": len(names),
        "ssim": regression_metrics(y_true[:, 0], y_pred[:, 0]),
        "psnr": regression_metrics(y_true[:, 1], y_pred[:, 1]),
    }
    # H5: report a bootstrap CI on each test R2. With n_test=133 the interval
    # is wide, and any ablation difference smaller than it is not a finding.
    for i, tgt in enumerate(("ssim", "psnr")):
        lo, hi = bootstrap_r2_ci(y_true[:, i], y_pred[:, i], n_boot=CNN_N_BOOTSTRAP)
        metrics[tgt]["r2_ci95"] = [round(lo, 4), round(hi, 4)]
    metrics["avg_R2_SSIM_PSNR_project_defined"] = combined_r2(
        metrics["ssim"]["r2"], metrics["psnr"]["r2"])
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{run_tag}] n_test={len(names)} "
          f"SSIM R²={metrics['ssim']['r2']:.4f} "
          f"CI95={metrics['ssim']['r2_ci95']} RMSE={metrics['ssim']['rmse']:.4f} | "
          f"PSNR R²={metrics['psnr']['r2']:.4f} "
          f"CI95={metrics['psnr']['r2_ci95']} RMSE={metrics['psnr']['rmse']:.4f} dB | "
          f"avgR²={metrics['avg_R2_SSIM_PSNR_project_defined']:.4f}")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run tag, e.g. hybrid_final")
    args = ap.parse_args()
    evaluate_run(args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
