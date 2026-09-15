"""Extraction of the 25 handcrafted image descriptors.

IMPORTANT CONTRACTS
-------------------
1. Input must be an RGB uint8 image (H, W, 3). OpenCV reads images as BGR,
   so callers MUST convert with cv2.cvtColor(img, cv2.COLOR_BGR2RGB) first.
   This is enforced (channel-order validation cannot be automated, so the
   requirement is documented and every caller converts explicitly).

2. Features are extracted from the PREPROCESSED image only. Reference images
   are used solely as SSIM/PSNR target sources, never as feature inputs.

3. Two correctness fixes relative to the draft implementation in the project
   notes (both change numeric values, hence features are re-extracted):
   (a) ``colorfulness``: the draft computed ``R - G`` and ``R + G`` on uint8
       arrays, where subtraction underflows (10 - 200 -> 66) and addition
       overflows (200 + 100 -> 44). Channels are now cast to float32 first.
       The *definition* (Hasler & Süsstrunk-style opponent std) is unchanged.
   (b) ``rms_contrast`` is algebraically identical to ``std`` (both are the
       standard deviation of gray values) and ``ASM = energy**2`` is a
       monotone transform of ``energy``. These are KEPT in the 25-feature set
       (to reproduce the documented redundancy analysis) and removed by the
       correlation filter — not silently dropped here.

Feature glossary (for the README):
- Canny edge detector: finds strong intensity discontinuities -> binary map.
- edge_density: fraction of pixels flagged as edges.
- SIFT (Scale-Invariant Feature Transform): local keypoints robust to scale
  and rotation. keypoint_density = #keypoints / image area.
- GLCM (Gray-Level Co-occurrence Matrix): spatial gray-level relationships
  used for texture properties. energy = textural uniformity;
  ASM (Angular Second Moment) = energy^2 under skimage's square-root form.
- entropy: Shannon entropy of gray-level distribution (complexity).
- laplacian_variance: focus/sharpness-related measure (Laplacian responds to
  high-frequency intensity change).
- glcm_variance: project-defined = variance of the normalised GLCM entries
  themselves (NOT the classic Haralick "sum of squares" texture variance).
  Kept under this documented definition for comparability with prior runs.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

from src.config import FEATURE_NAMES_25

__all__ = ["extract_features", "FEATURE_NAMES_25"]

_sift = None  # created lazily once (creating it per image is wasteful)


def _get_sift():
    """Return a shared SIFT detector, or raise a clear error if unavailable."""
    global _sift
    if _sift is None:
        try:
            _sift = cv2.SIFT_create()
        except AttributeError as e:  # pragma: no cover - build without SIFT
            raise RuntimeError(
                "cv2.SIFT_create() is unavailable in this OpenCV build. "
                "Install opencv-python>=4.5 (which bundles SIFT) to reproduce "
                "the documented keypoint_density definition."
            ) from e
    return _sift


def extract_features(image_rgb: np.ndarray) -> dict:
    """Extract the 25 handcrafted features from one RGB uint8 image."""
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("extract_features() expects an HxWx3 RGB image.")
    if image_rgb.dtype != np.uint8:
        raise ValueError("extract_features() expects a uint8 image.")

    features: dict[str, float] = {}

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Float copies for any arithmetic that could under/overflow in uint8.
    Rf = image_rgb[:, :, 0].astype(np.float32)
    Gf = image_rgb[:, :, 1].astype(np.float32)
    Bf = image_rgb[:, :, 2].astype(np.float32)

    # ================= STATISTICAL =================
    features["mean"] = float(np.mean(gray))
    features["std"] = float(np.std(gray))
    features["variance"] = float(np.var(gray))
    features["entropy"] = float(shannon_entropy(gray))
    features["dynamic_range"] = float(np.max(gray) - np.min(gray))
    features["rms_contrast"] = float(np.sqrt(np.mean((gray - np.mean(gray)) ** 2)))

    # ================= COLOUR =================
    features["mean_red"] = float(np.mean(Rf))
    features["mean_green"] = float(np.mean(Gf))
    features["mean_blue"] = float(np.mean(Bf))

    rg = Rf - Gf
    yb = 0.5 * (Rf + Gf) - Bf
    features["colorfulness"] = float(np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2))
    features["red_ratio"] = float(
        np.mean(Rf) / (np.mean(Gf) + np.mean(Bf) + 1e-10)
    )
    features["mean_saturation"] = float(np.mean(hsv[:, :, 1]))
    features["mean_value"] = float(np.mean(hsv[:, :, 2]))

    # ================= GLCM TEXTURE =================
    glcm = graycomatrix(
        gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )
    features["contrast"] = float(graycoprops(glcm, "contrast")[0, 0])
    features["correlation"] = float(graycoprops(glcm, "correlation")[0, 0])
    features["energy"] = float(graycoprops(glcm, "energy")[0, 0])
    features["homogeneity"] = float(graycoprops(glcm, "homogeneity")[0, 0])
    features["ASM"] = float(features["energy"] ** 2)
    features["dissimilarity"] = float(graycoprops(glcm, "dissimilarity")[0, 0])
    features["glcm_entropy"] = float(-np.sum(glcm * np.log2(glcm + 1e-10)))
    features["glcm_variance"] = float(np.var(glcm))

    # ================= EDGE / SHARPNESS =================
    edges = cv2.Canny(gray, 100, 200)
    features["edge_density"] = float(np.mean(edges > 0))

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    features["gradient"] = float(np.mean(np.sqrt(gx**2 + gy**2)))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features["laplacian_variance"] = float(np.var(laplacian))

    keypoints, _ = _get_sift().detectAndCompute(gray, None)
    features["keypoint_density"] = float(
        len(keypoints) / (gray.shape[0] * gray.shape[1])
    )

    # Guard: every feature must be finite; fail loudly, never silently impute.
    for name in FEATURE_NAMES_25:
        v = features[name]
        if not np.isfinite(v):
            raise ValueError(f"Non-finite value for feature '{name}': {v}")
    return features
