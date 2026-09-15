"""Regression metrics with honest, explicit naming.

- r2 / rmse / mae: standard regression metrics (test/val only, never train).
- pearson_r: supplementary predicted-vs-actual linear correlation.
- combined_r2(): PROJECT-DEFINED summary = mean of SSIM R² and PSNR R².
  It is scale-free (both inputs are R²) so averaging is mathematically
  coherent, but it is NOT a standard image-quality metric. Every table that
  uses it must label it exactly "avg_R2_SSIM_PSNR (project-defined)".
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "n": int(len(y_true)),
    }


def combined_r2(ssim_r2: float, psnr_r2: float) -> float:
    """Project-defined summary: average held-out R² across SSIM and PSNR."""
    return float((ssim_r2 + psnr_r2) / 2.0)
