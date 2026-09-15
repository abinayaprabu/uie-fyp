"""Classical (non-CNN) underwater image enhancement pipeline.

Flow (fixed order):
    RAW (BGR, as read by OpenCV)
      -> aspect-preserving resize to width 600 px (INTER_AREA)
      -> Gray-World white balance (per-image gains, no cross-image fitting,
         hence no information leakage across the dataset)
      -> CLAHE on the L-channel of CIELAB (clipLimit=2.5, tiles 8x8)
      -> bilateral filtering (d=9, sigmaColor=75, sigmaSpace=75)
      -> adaptive gamma correction driven by the image's own mean gray value
      -> PREPROCESSED (BGR uint8, same convention as cv2.imread)

Two implementation choices are documented here because the original project
notes only gave parameter values, not exact definitions:

1. CLAHE is applied to the L (lightness) channel of CIELAB and then merged
   back. This enhances local contrast without shifting colours. Applying
   CLAHE independently to R/G/B would distort colour balance and is NOT used.

2. Adaptive gamma:  gamma = clip(mean_gray / 128, 0.5, 2.0).
   Dark images (mean < 128) get gamma < 1 (brightened), bright images get
   gamma > 1 (pulled back), mid-tone images stay near gamma = 1. The rule is
   continuous, deterministic and uses only the image itself.

The reference image is resized with the *identical* geometric transform
(same width, same interpolation) before SSIM/PSNR computation so that both
images being compared share one documented resampling step.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.config import (
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    GAMMA_MAX,
    GAMMA_MIN,
    RESIZE_WIDTH,
)


def resize_to_width(image_bgr: np.ndarray, width: int = RESIZE_WIDTH) -> np.ndarray:
    """Aspect-preserving resize to a fixed width (INTER_AREA for downsampling)."""
    h, w = image_bgr.shape[:2]
    if w == width:
        return image_bgr
    scale = width / w
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (width, new_h), interpolation=cv2.INTER_AREA)


def gray_world_white_balance(image_bgr: np.ndarray) -> np.ndarray:
    """Gray-World white balance: scale each channel so channel means are equal.

    Gains are computed from the image itself (mean over channels / channel
    mean), so no dataset-level statistics are involved.
    """
    img = image_bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    means = np.where(means < 1e-6, 1e-6, means)  # guard against black channels
    gray = means.mean()
    gains = gray / means
    balanced = np.clip(img * gains, 0, 255).astype(np.uint8)
    return balanced


def apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the L-channel of CIELAB (colour-preserving local contrast)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def apply_bilateral(image_bgr: np.ndarray) -> np.ndarray:
    """Edge-preserving smoothing (removes noise, keeps edges)."""
    return cv2.bilateralFilter(
        image_bgr, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
    )


def adaptive_gamma(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Gamma correction with gamma = clip(mean_gray/128, 0.5, 2.0).

    Returns the corrected image and the gamma value used (for logging).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gamma = float(np.clip(gray.mean() / 128.0, GAMMA_MIN, GAMMA_MAX))
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(image_bgr, lut), gamma


def enhance(raw_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Run the full classical enhancement pipeline on one image.

    Args:
        raw_bgr: raw underwater image in OpenCV BGR format (H, W, 3) uint8.

    Returns:
        (preprocessed_bgr, info) where info records gamma and output shape.
    """
    if raw_bgr is None or raw_bgr.ndim != 3 or raw_bgr.shape[2] != 3:
        raise ValueError("enhance() expects an HxWx3 colour image.")
    img = resize_to_width(raw_bgr)
    img = gray_world_white_balance(img)
    img = apply_clahe(img)
    img = apply_bilateral(img)
    img, gamma = adaptive_gamma(img)
    return img, {"gamma": gamma, "shape": img.shape}


def resize_reference_like_preprocessed(reference_bgr: np.ndarray) -> np.ndarray:
    """Apply the identical geometric resize to a reference image.

    UIEB raw/reference pairs share dimensions, so resizing the reference to
    width 600 with INTER_AREA reproduces exactly the geometric transform the
    raw image underwent. This single documented resampling step is required
    for pixel-aligned SSIM/PSNR.
    """
    return resize_to_width(reference_bgr)
