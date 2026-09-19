"""Full-reference image quality metrics — the project's SINGLE definition.

``scripts/build_feature_dataset.py::compute_ssim_psnr`` created the SSIM/PSNR
labels that every model in this repository is trained and judged against. The
enhancement work must score its output with *exactly* the same definition, or a
comparison against the classical baseline (test mean SSIM 0.7636, PSNR
17.089 dB) would be meaningless.

Rather than import from a script, the definition is restated here verbatim and
then PROVEN equivalent: ``scripts/verify_enhancement.py`` recomputes all 890
pairs with this module and requires bit-identical agreement with the committed
``results/feature/feature_quality_dataset.csv``. If that check passes, the two
definitions are the same function, and the frozen pipeline has not been touched.

Definitions (scikit-image defaults, stated explicitly so they cannot drift):
  SSIM : ``structural_similarity(a, b, channel_axis=2, data_range=255)``
         -> uniform 7x7 window (``gaussian_weights=False``), K1=0.01, K2=0.03,
            computed on uint8 RGB.
  PSNR : ``peak_signal_noise_ratio(ref, test, data_range=255)``
         -> NOTE the argument order: the REFERENCE comes first.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_ssim_psnr(test_bgr: np.ndarray, ref_bgr: np.ndarray) -> tuple[float, float]:
    """SSIM and PSNR of ``test_bgr`` against ``ref_bgr``. Both BGR uint8,
    identical shapes (the caller aligns them with the frozen resize)."""
    if test_bgr.shape != ref_bgr.shape:
        raise ValueError(f"shape mismatch: test {test_bgr.shape} vs "
                         f"reference {ref_bgr.shape}")
    if test_bgr.dtype != np.uint8 or ref_bgr.dtype != np.uint8:
        raise ValueError("SSIM/PSNR here are defined on uint8 with data_range=255")
    a = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    ssim = float(structural_similarity(a, b, channel_axis=2, data_range=255))
    psnr = float(peak_signal_noise_ratio(b, a, data_range=255))
    return ssim, psnr


def psnr_from_scratch(test: np.ndarray, ref: np.ndarray,
                      data_range: float = 255.0) -> float:
    """PSNR implemented directly from its formula, as an independent cross-check
    on scikit-image. MSE is over ALL pixels and ALL channels."""
    t = np.asarray(test, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if t.shape != r.shape:
        raise ValueError(f"shape mismatch {t.shape} vs {r.shape}")
    mse = float(np.mean((t - r) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim_from_scratch(test: np.ndarray, ref: np.ndarray, win: int = 7,
                      data_range: float = 255.0,
                      k1: float = 0.01, k2: float = 0.03) -> float:
    """Mean SSIM over channels with a UNIFORM win x win window — an independent
    reimplementation used only to cross-check scikit-image, not for reporting.

    Matches ``structural_similarity`` defaults (``gaussian_weights=False``).
    Computed in float64 with the same unbiased-by-filter convention, so agreement
    is expected to ~1e-10 rather than exactly.
    """
    from scipy.ndimage import uniform_filter

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    pad = (win - 1) // 2           # scikit-image trims this border; so must we
    # scikit-image defaults to use_sample_covariance=True, i.e. the variances
    # and covariance are scaled by NP/(NP-1) with NP = win**2 per 2-D channel.
    # Omitting this factor leaves a systematic ~1e-3 offset; including it makes
    # the two implementations agree to ~1e-12 (verified on UIEB pairs).
    cov_norm = (win ** 2) / (win ** 2 - 1)
    total = 0.0
    n_ch = np.asarray(test).shape[2]
    for ch in range(n_ch):
        x = np.asarray(test)[:, :, ch].astype(np.float64)
        y = np.asarray(ref)[:, :, ch].astype(np.float64)
        ux = uniform_filter(x, win)
        uy = uniform_filter(y, win)
        uxx = uniform_filter(x * x, win)
        uyy = uniform_filter(y * y, win)
        uxy = uniform_filter(x * y, win)
        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)
        num = (2 * ux * uy + c1) * (2 * vxy + c2)
        den = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
        ssim_map = num / den
        total += float(np.mean(ssim_map[pad:-pad, pad:-pad], dtype=np.float64))
    return total / n_ch
