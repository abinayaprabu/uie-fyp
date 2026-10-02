"""No-reference underwater image quality metrics: UIQM and UCIQE.

WHY THESE TWO
-------------
PSNR and SSIM need a reference image. UIQM and UCIQE do not: they score an
image on its own, which is exactly what the UIEB "challenging-60" set and real
deployments need. They are reported alongside PSNR/SSIM as COMPLEMENTARY
evidence (see `docs/metrics_explanation.md`), never as a replacement.

A WARNING THAT MUST BE REPEATED IN EVERY REPORT
-----------------------------------------------
Neither metric knows what the true scene looked like. An image with more
colour and contrast can score higher while being FURTHER from the ground
truth. Their absolute values also depend on implementation details (block
sizes, PLIP constant, colour-space conventions), so numbers must never be
compared against values printed in other papers -- only against other methods
evaluated with THIS file.

DEFINITIONS USED HERE (documented so they cannot drift)
-------------------------------------------------------
UIQM (Panetta, Gao & Agaian, 2016, "Human-Visual-System-Inspired Underwater
Image Quality Measures", IEEE J. Oceanic Eng.):
    UIQM = 0.0282 * UICM + 0.2953 * UISM + 3.5753 * UIConM
  * UICM (colourfulness), opponent channels RG = R - G and YB = (R+G)/2 - B,
    with ALPHA-TRIMMED statistics (10% cut from each tail, alpha_L = alpha_R =
    0.1):
        UICM = -0.0268 * sqrt(mu_RG^2 + mu_YB^2)
               + 0.1586 * sqrt(sig_RG^2 + sig_YB^2)
  * UISM (sharpness): Sobel edge magnitude per RGB channel, then the
    enhancement-measure EME over 8x8 blocks of each edge map:
        EME = (1/(k1*k2)) * sum 20*log10(block_max / block_min)
        UISM = 0.299*EME(R) + 0.587*EME(G) + 0.114*EME(B)
  * UIConM (contrast): logAMEE on the mean intensity image, using PLIP
    (parameterised logarithmic image processing) operators with the standard
    constant 1026 for 8-bit data:
        UIConM = (1/(k1*k2)) * sum_c [ c * ln(c) ],
        c = PLIP_sub(Imax, Imin) / PLIP_add(Imax, Imin)
        PLIP_add(a, b) = a + b - a*b/1026
        PLIP_sub(a, b) = (a - b) / (1 - b/1026)

UCIQE (Yang & Sowmya, 2015, "An Underwater Color Image Quality Evaluation
Metric", IEEE Trans. Image Processing):
    UCIQE = 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s
    in CIELAB: chroma = sqrt(a*^2 + b*^2);
      sigma_c = standard deviation of chroma over all pixels,
      con_l   = P99(L*) - P1(L*)   (difference of the top and bottom 1%
                 of luminance, i.e. its robust dynamic range),
      mu_s    = mean of per-pixel saturation chroma / L*.

All functions take an RGB uint8 image (H, W, 3) and return a float. The
sanity tests live in ``scripts/test_nr_metrics.py`` (flat images -> 0;
blurring lowers UISM; removing colour lowers UICM/UCIQE; the vectorised code
here is checked against the plain double loops written out above).

A PROPERTY OF THE PUBLISHED UIConM THAT MUST BE STATED, NOT HIDDEN
------------------------------------------------------------------
The published logAMEE term is ``c * ln(c)`` with ``c`` in (0, 1), which is
NON-MONOTONIC: it is ~0 for c -> 0, falls to -1/e at c = 1/e, and rises back
toward 0 as c -> 1. It is therefore NOT a monotone "more contrast is better"
score, and its absolute value is notoriously implementation-dependent. This
project therefore (a) computes it exactly as published and identically for
every method, (b) reports it only as a component of UIQM, and (c) never uses
UIQM/UCIQE as proof that one image is closer to the ground truth. PSNR/SSIM
(full reference) remain the primary evidence.
"""
from __future__ import annotations

import numpy as np
from skimage import color

# --- UIQM constants (Panetta et al. 2016) ---------------------------------
UIQM_C1, UIQM_C2, UIQM_C3 = 0.0282, 0.2953, 3.5753
UICM_LAMBDA1, UICM_LAMBDA2 = -0.0268, 0.1586
ALPHA_L = ALPHA_R = 0.1          # alpha-trimming fraction per tail
EME_BLOCK = 8                    # 8x8 blocks for EME and logAMEE
UISM_W = (0.299, 0.587, 0.114)   # R, G, B weights (human visual system)
PLIP_LAMBDA = 1026.0             # standard PLIP constant for 8-bit images

# --- UCIQE constants (Yang & Sowmya 2015) ---------------------------------
UCIQE_C1, UCIQE_C2, UCIQE_C3 = 0.4680, 0.2745, 0.2576


def _check(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got {a.shape}")
    if a.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {a.dtype}")
    return a.astype(np.float64)


def uicm(rgb: np.ndarray) -> float:
    """Underwater Image Colourfulness Measure (alpha-trimmed opponent stats)."""
    a = _check(rgb)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    rg = (r - g).ravel()
    yb = ((r + g) / 2.0 - b).ravel()
    out = []
    for ch in (rg, yb):
        ch = np.sort(ch)
        k = ch.size
        lo = int(ALPHA_L * k + 1)
        hi = int(k * (1.0 - ALPHA_R))
        trimmed = ch[lo:hi]
        n = trimmed.size
        mu = float(trimmed.sum() / n)
        sigma = float(np.sqrt(np.sum((trimmed - mu) ** 2) / n))
        out.append((mu, sigma))
    (mu_rg, sig_rg), (mu_yb, sig_yb) = out
    return (UICM_LAMBDA1 * np.sqrt(mu_rg ** 2 + mu_yb ** 2)
            + UICM_LAMBDA2 * np.sqrt(sig_rg ** 2 + sig_yb ** 2))


def _blocks(img2d: np.ndarray, block: int = EME_BLOCK) -> np.ndarray:
    """Split into non-overlapping blocks -> array (k1, k2, block, block).

    Vectorised form of the double loop over blocks; the arithmetic is
    identical, it is just faster on full-resolution images.
    """
    h, w = img2d.shape
    k1, k2 = h // block, w // block
    if k1 == 0 or k2 == 0:
        return np.empty((0, 0, block, block))
    cropped = img2d[:k1 * block, :k2 * block]
    return cropped.reshape(k1, block, k2, block).transpose(0, 2, 1, 3)


def _eme(img2d: np.ndarray, block: int = EME_BLOCK) -> float:
    """Enhancement Measure Estimation: mean 20*log10(max/min) over blocks.

    Vectorised: the image is cropped to whole blocks and reshaped into a
    (k1, k2, block*block) array; the arithmetic is identical to the double
    loop over blocks in the published definition.
    """
    h, w = img2d.shape
    k1, k2 = h // block, w // block
    if k1 == 0 or k2 == 0:
        return 0.0
    b = (img2d[:k1 * block, :k2 * block]
         .reshape(k1, block, k2, block)
         .transpose(0, 2, 1, 3)
         .reshape(k1, k2, block * block))
    mx = b.max(axis=2)
    # Guard against log(0): lowest expressible level of the block statistic.
    mn = np.maximum(b.min(axis=2), 1.0)
    valid = mx > 0
    total = float(np.sum(20.0 * np.log10(mx[valid] / mn[valid]))) if valid.any() else 0.0
    # divide by the TOTAL number of blocks: an all-zero block contributes zero
    return total / (k1 * k2)


def uism(rgb: np.ndarray) -> float:
    """Underwater Image Sharpness Measure: EME of the Sobel edge maps."""
    import cv2

    a = _check(rgb)
    score = 0.0
    for c, weight in enumerate(UISM_W):
        ch = a[..., c]
        gx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.abs(gx) + np.abs(gy)       # L1 gradient magnitude
        score += weight * _eme(edge)
    return float(score)


def _plip_add(a: float, b: float, lam: float = PLIP_LAMBDA) -> float:
    return a + b - (a * b) / lam


def _plip_sub(a: float, b: float, lam: float = PLIP_LAMBDA) -> float:
    denom = 1.0 - b / lam
    if denom <= 1e-12:
        return 0.0
    return (a - b) / denom


def uiconm(rgb: np.ndarray) -> float:
    """Underwater Image Contrast Measure: logAMEE with PLIP operators.

    Vectorised over blocks; identical arithmetic to the scalar form
    (``_plip_sub`` / ``_plip_add`` are applied elementwise with the same
    guards, and skipped blocks count as zero).
    """
    a = _check(rgb)
    gray = a.mean(axis=2)                    # (R+G+B)/3
    h, w = gray.shape
    k1, k2 = h // EME_BLOCK, w // EME_BLOCK
    if k1 == 0 or k2 == 0:
        return 0.0
    b = (gray[:k1 * EME_BLOCK, :k2 * EME_BLOCK]
         .reshape(k1, EME_BLOCK, k2, EME_BLOCK)
         .transpose(0, 2, 1, 3)
         .reshape(k1, k2, EME_BLOCK * EME_BLOCK))
    mx = b.max(axis=2)
    mn = b.min(axis=2)
    den_sub = 1.0 - mn / PLIP_LAMBDA
    num = np.where(den_sub > 1e-12,
                   (mx - mn) / np.where(den_sub > 1e-12, den_sub, 1.0), 0.0)
    den_add = mx + mn - (mx * mn) / PLIP_LAMBDA
    c = np.where(den_add > 1e-12,
                 num / np.where(den_add > 1e-12, den_add, 1.0), 0.0)
    vals = np.where(c > 0, c * np.log(np.where(c > 0, c, 1.0)), 0.0)
    return float(vals.sum() / (k1 * k2))


def uiqm(rgb: np.ndarray) -> float:
    """The combined UIQM score (higher = more colour/sharpness/contrast)."""
    return float(UIQM_C1 * uicm(rgb) + UIQM_C2 * uism(rgb) + UIQM_C3 * uiconm(rgb))


def uiqm_components(rgb: np.ndarray) -> tuple[float, float, float]:
    """(UICM, UISM, UIConM) — useful for understanding a surprising UIQM."""
    return uicm(rgb), uism(rgb), uiconm(rgb)


def uciqe(rgb: np.ndarray) -> float:
    """Underwater Colour Image Quality Evaluation (higher = more colour/contrast)."""
    a = _check(rgb)
    lab = color.rgb2lab(a / 255.0)           # D65; L in [0,100], a,b in [-128,127]
    L = lab[..., 0]
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    sigma_c = float(chroma.std())            # population standard deviation
    con_l = float(np.percentile(L, 99) - np.percentile(L, 1))
    # Per-pixel saturation as chroma / luminance; L is clipped away from 0 to
    # keep the ratio finite (a documented convention, applied to every method
    # identically so comparisons stay valid).
    saturation = chroma / np.clip(L, 1e-6, None)
    mu_s = float(saturation.mean())
    return float(UCIQE_C1 * sigma_c + UCIQE_C2 * con_l + UCIQE_C3 * mu_s)


def uciqe_components(rgb: np.ndarray) -> tuple[float, float, float]:
    """(sigma_c, con_l, mu_s) for UCIQE."""
    a = _check(rgb)
    lab = color.rgb2lab(a / 255.0)
    L = lab[..., 0]
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    sigma_c = float(chroma.std())
    con_l = float(np.percentile(L, 99) - np.percentile(L, 1))
    mu_s = float((chroma / np.clip(L, 1e-6, None)).mean())
    return sigma_c, con_l, mu_s
