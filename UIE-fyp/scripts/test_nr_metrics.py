#!/usr/bin/env python3
"""Sanity tests for the no-reference metrics in ``src/nr_metrics.py``.

UIQM and UCIQE are the two metrics used in the final results table that do NOT
need a ground-truth reference image, so they must be trustworthy before they
are reported next to PSNR/SSIM. This script proves three things:

  1. the fast (vectorised) implementation is numerically identical to the
     plain double-loop definition written out in the module docstring;
  2. the scores behave as documented on controlled images (flat images score
     0, blurring lowers sharpness, removing colour lowers colourfulness);
  3. the module's own limitations are true, not assumed: the published
     logAMEE ("UIConM") term is NON-monotonic, and neither metric knows where
     the ground truth is -- so they stay a complementary signal only.

Run:  python scripts/test_nr_metrics.py      (exits 1 if a check fails)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import nr_metrics as M  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def loop_eme(img2d: np.ndarray, block: int = 8) -> float:
    """Plain double loop, exactly the formula in the module docstring."""
    h, w = img2d.shape
    k1, k2 = h // block, w // block
    if k1 == 0 or k2 == 0:
        return 0.0
    total = 0.0
    for i in range(k1):
        for j in range(k2):
            blk = img2d[i * block:(i + 1) * block, j * block:(j + 1) * block]
            mx = float(blk.max())
            if mx <= 0:                     # all-zero block contributes zero
                continue
            mn = max(float(blk.min()), 1.0)  # documented guard against log(0)
            total += 20.0 * np.log10(mx / mn)
    return total / (k1 * k2)


def loop_uiconm(rgb: np.ndarray) -> float:
    """Plain double loop over the PLIP logAMEE term."""
    gray = rgb.astype(np.float64).mean(axis=2)
    h, w = gray.shape
    k1, k2 = h // 8, w // 8
    total = 0.0
    for i in range(k1):
        for j in range(k2):
            blk = gray[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            mx, mn = float(blk.max()), float(blk.min())
            den_sub = 1.0 - mn / M.PLIP_LAMBDA
            num = (mx - mn) / den_sub if den_sub > 1e-12 else 0.0
            den_add = mx + mn - (mx * mn) / M.PLIP_LAMBDA
            if den_add <= 1e-12:
                continue
            c = num / den_add
            if c > 0:
                total += c * np.log(c)
    return total / (k1 * k2)


def flat(v: int, shape=(64, 64)) -> np.ndarray:
    return np.full((*shape, 3), v, np.uint8)


def sample_images() -> list[tuple[str, np.ndarray]]:
    """Up to three real PNGs if they are on disk, else synthetic images."""
    out: list[tuple[str, np.ndarray]] = []
    folder = REPO / "dataset" / "enhanced-test"
    if folder.is_dir():
        for p in sorted(folder.glob("*.png"))[:3]:
            bgr = cv2.imread(str(p))
            if bgr is not None:
                out.append((p.name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    if len(out) < 3:
        rng = np.random.default_rng(42)
        for k in range(3 - len(out)):
            base = rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)
            out.append((f"synthetic_{k}", cv2.GaussianBlur(base, (5, 5), 0)))
    return out


def main() -> int:
    print("=" * 78)
    print("SANITY TESTS -- no-reference metrics (src/nr_metrics.py)")
    print("=" * 78)

    print("\n[1] flat images must score ~0 (no structure, no colour, no contrast)")
    # UCIQE tolerance: skimage's rgb2lab leaves ~1e-3 chroma residue on exactly
    # grey input, i.e. ~1e-5 in the final score -- six orders of magnitude below
    # a real image (~25). A genuine bug (e.g. unclipped saturation) is far
    # larger, so 1e-3 still detects one.
    for label, im in [("grey 128", flat(128)), ("white", flat(255)), ("black", flat(0))]:
        u, q = M.uiqm(im), M.uciqe(im)
        check(f"{label}: UIQM=0, UCIQE=0", abs(u) < 1e-9 and abs(q) < 1e-3,
              f"UIQM={u:.3e} UCIQE={q:.3e} (flat-image float residue)")

    print("\n[2] the module rejects silently-wrong input types")
    try:
        M.uiqm(flat(128).astype(np.float32))
        check("float input raises ValueError", False, "no exception")
    except ValueError:
        check("float input raises ValueError", True)

    print("\n[3] vectorised == plain double loop (exact arithmetic, same guards)")
    worst_e = worst_c = 0.0
    images = sample_images()
    for name, rgb in images:
        for c in range(3):
            g = rgb[..., c].astype(np.float64)
            edge = (np.abs(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3))
                    + np.abs(cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)))
            worst_e = max(worst_e, abs(M._eme(edge) - loop_eme(edge)))
        worst_c = max(worst_c, abs(M.uiconm(rgb) - loop_uiconm(rgb)))
    check("EME matches (<=1e-9)", worst_e <= 1e-9, f"max |diff| = {worst_e:.2e}")
    check("UIConM matches (<=1e-9)", worst_c <= 1e-9, f"max |diff| = {worst_c:.2e}")

    print("\n[4] directional behaviour on a real image")
    name, img = images[0]
    blur = cv2.GaussianBlur(img, (15, 15), 4.0)
    grey = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    print(f"    {name}: UIQM={M.uiqm(img):.4f} UCIQE={M.uciqe(img):.4f}")
    print(f"    blurred:  UIQM={M.uiqm(blur):.4f} UCIQE={M.uciqe(blur):.4f}")
    print(f"    greyscale: UIQM={M.uiqm(grey):.4f} UCIQE={M.uciqe(grey):.4f}")
    check("blurring lowers UISM (sharpness)", M.uism(blur) < M.uism(img),
          f"{M.uism(blur):.3f} < {M.uism(img):.3f}")
    check("removing colour lowers UICM (colourfulness)", M.uicm(grey) < M.uicm(img),
          f"{M.uicm(grey):.3f} < {M.uicm(img):.3f}")
    check("UCIQE uses colour (grey image scores lower)", M.uciqe(grey) < M.uciqe(img),
          f"{M.uciqe(grey):.3f} < {M.uciqe(img):.3f}")

    print("\n[5] UIQM = 0.0282*UICM + 0.2953*UISM + 3.5753*UIConM (components agree)")
    uicm, uism, uiconm = M.uiqm_components(img)
    total = M.UIQM_C1 * uicm + M.UIQM_C2 * uism + M.UIQM_C3 * uiconm
    check("component decomposition", abs(total - M.uiqm(img)) < 1e-9,
          f"|diff| = {abs(total - M.uiqm(img)):.2e}")

    print("\n[6] documented limitations are real (do not hide them)")
    low_contrast = np.clip(128 + (img.astype(np.float64) - 128) * 0.35, 0, 255).astype(np.uint8)
    c_orig, c_low = M.uiconm(img), M.uiconm(low_contrast)
    print(f"    UIConM: original {c_orig:.4f} | low-contrast {c_low:.4f} "
          f"({'higher' if c_low > c_orig else 'lower'} for less contrast)")
    check("UIConM is not monotone, as documented", c_low > c_orig,
          "the published c*ln(c) term rises as c falls below 1/e; "
          "never read UIConM alone as 'more contrast is better'")
    check("deterministic on repeated calls",
          M.uiqm(img) == M.uiqm(img) and M.uciqe(img) == M.uciqe(img))

    print("\n[7] speed at full resolution (used inside the evaluation loop)")
    t0 = time.time()
    for _, im in images:
        M.uiqm(im)
        M.uciqe(im)
    per_image = (time.time() - t0) / len(images)
    check("faster than 1.5 s/image", per_image < 1.5, f"{per_image:.3f} s/image")

    print("\n" + "=" * 78)
    if failures:
        print(f"RESULT: {len(failures)} CHECK(S) FAILED -> {failures}")
        return 1
    print("RESULT: ALL NR-METRIC CHECKS PASSED")
    print("Reminder for the report: UIQM/UCIQE are implementation-dependent "
          "no-reference scores;\ncompare them only between methods measured by "
          "THIS file, never against other papers' tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
