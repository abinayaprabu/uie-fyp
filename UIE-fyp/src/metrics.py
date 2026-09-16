"""Regression metrics with honest, explicit naming.

- r2 / rmse / mae: standard regression metrics (test/val only, never train).
- pearson_r: supplementary predicted-vs-actual linear correlation.
- bootstrap_r2_ci(): percentile bootstrap CI for R2 - REQUIRED when reporting,
  because n_test=133 makes the interval roughly +/-0.15 wide.
- paired_wilcoxon(): paired test on per-sample squared errors, the correct way
  to ask whether two models genuinely differ on the SAME test images.
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


def bootstrap_r2_ci(y_true, y_pred, n_boot: int = 4000, seed: int = 0,
                    alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for R² (H5: n_test=133 gives a wide interval).

    Resamples prediction/target PAIRS with replacement so the CI reflects
    sampling variability of the test set, not of the model. Essential here:
    with 133 test images the interval is roughly +/-0.15 on R², so differences
    between ablations of 0.02-0.05 cannot be claimed without it.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yt = y_true[i]
        if yt.var() == 0:      # degenerate resample; R² undefined
            continue
        vals.append(r2_score(yt, y_pred[i]))
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def paired_wilcoxon(err_a, err_b) -> float:
    """Two-sided Wilcoxon signed-rank p-value on per-sample SQUARED errors.

    Used to ask whether model A's test error genuinely differs from model B's
    on the SAME test images (paired), which is the correct test for the
    ablation comparisons. Returns 1.0 when the errors are identical.
    """
    from scipy.stats import wilcoxon

    a = np.asarray(err_a, dtype=float)
    b = np.asarray(err_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0
