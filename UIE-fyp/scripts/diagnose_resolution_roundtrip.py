#!/usr/bin/env python3
"""Diagnostic: how much of the feature-guided model's test gap is the 224px round trip?

The proposed model works at 224x224 (letterboxed) and its output is resized
back to the reference geometry before scoring. The existing U-Net baseline was
trained and evaluated at full resolution. This script measures four controls on
the SAME 133 sealed test images, so the gap can be attributed instead of
guessed:

  A. classical            : preprocessed image vs reference (the baseline row)
  B. identity round trip  : preprocessed -> letterbox(224) -> crop -> resize back
  C. model @ its own size : model output (224) vs letterboxed reference (224)
  D. model round trip     : the actual enhanced PNG (what the table reports)

B isolates the information cost of working at 224 and coming back. If B is much
lower than A, part of the gap is a resolution handicap; the rest is the model.
C vs D shows how much the final resize costs the model.

Writes results/metrics/resolution_roundtrip.csv and prints the summary.

Usage:  python scripts/diagnose_resolution_roundtrip.py --run enh224_featguided
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn.dataset import letterbox  # noqa: E402
from cnn.dataset_pairs import aligned_reference, split_ids, to_tensor, to_uint8  # noqa: E402
from cnn.feature_guided.enhance import letterbox_region, load_model, scale_features  # noqa: E402
from src.config import (  # noqa: E402
    ENHANCEMENT_RESULTS_DIR, FEATURE_RESULTS_DIR, HYBRID_INPUT_SIZE,
    PREPROCESSED_DIR, RESULTS_DIR,
)
from src.iqa import compute_ssim_psnr  # noqa: E402

OUT = RESULTS_DIR / "metrics" / "resolution_roundtrip.csv"
QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="enh224_featguided")
    args = ap.parse_args()

    model, ckpt = load_model(args.run)
    names = ckpt["feature_names"]
    q = pd.read_csv(QUALITY_CSV).set_index("image_name")
    size = HYBRID_INPUT_SIZE

    rows = []
    for name in split_ids("test"):
        ref = aligned_reference(name)                     # BGR uint8, full res
        pre = cv2.imread(str(PREPROCESSED_DIR / name), cv2.IMREAD_COLOR)
        if pre is None or pre.shape != ref.shape:
            raise ValueError(f"bad preprocessed image {name}")

        # --- B: identity round trip (no network at all) --------------------
        # preprocessed -> letterbox 224 -> crop the image area -> resize back.
        # This is exactly the geometry the model's output goes through, applied
        # to the input image itself, so it isolates the resolution cost.
        lb = letterbox(cv2.cvtColor(pre, cv2.COLOR_BGR2RGB), size)
        h, w = pre.shape[:2]
        nh, nw, top, left = letterbox_region(h, w, size)
        crop_rgb = lb[top:top + nh, left:left + nw]
        ident = cv2.resize(crop_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        s_id, p_id = compute_ssim_psnr(cv2.cvtColor(ident, cv2.COLOR_RGB2BGR), ref)

        # --- C: model output at 224 vs letterboxed reference ----------------
        feats = scale_features(ckpt, names, q.loc[name].to_dict())
        with torch.no_grad():
            x = to_tensor(lb).unsqueeze(0)
            out224 = model(x, torch.from_numpy(feats))[0]          # (3,224,224)
        ref_lb = letterbox(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB), size)
        m224 = to_uint8(out224)                                    # RGB uint8
        # Compare INSIDE the image area only: the black letterbox bars are
        # identical in both images and would inflate SSIM if left in.
        m_crop = m224[top:top + nh, left:left + nw]
        r_crop = ref_lb[top:top + nh, left:left + nw]
        from skimage.metrics import structural_similarity as ssim_fn, peak_signal_noise_ratio as psnr_fn
        s_224 = float(ssim_fn(m_crop, r_crop, channel_axis=2, data_range=255))
        p_224 = float(psnr_fn(r_crop, m_crop, data_range=255))

        # --- D: the actual enhanced image that the table reports ------------
        enh = cv2.imread(str(ENHANCEMENT_RESULTS_DIR / args.run / "enhanced" / name),
                         cv2.IMREAD_COLOR)
        s_rt, p_rt = compute_ssim_psnr(enh, ref)

        # --- A: classical ---------------------------------------------------
        s_cl, p_cl = compute_ssim_psnr(pre, ref)

        rows.append({"image_name": name,
                     "ssim_classical": s_cl, "psnr_classical": p_cl,
                     "ssim_identity_roundtrip": s_id, "psnr_identity_roundtrip": p_id,
                     "ssim_model_at224": s_224, "psnr_model_at224": p_224,
                     "ssim_model_roundtrip": s_rt, "psnr_model_roundtrip": p_rt})

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    def m(col):
        return df[col].mean()

    print("=" * 78)
    print(f"RESOLUTION / ROUND-TRIP DIAGNOSTIC (n={len(df)}, run {args.run})")
    print("=" * 78)
    print(f"{'control':<44}{'SSIM':>9}{'PSNR':>9}")
    for label, sc, pc in [
        ("A  classical pipeline (baseline)", "ssim_classical", "psnr_classical"),
        ("B  identity: pre -> 224 -> pre geometry", "ssim_identity_roundtrip",
         "psnr_identity_roundtrip"),
        ("C  model output at 224 vs ref (image area)", "ssim_model_at224", "psnr_model_at224"),
        ("D  model output resized to ref (reported)", "ssim_model_roundtrip",
         "psnr_model_roundtrip"),
    ]:
        print(f"{label:<44}{m(sc):>9.4f}{m(pc):>9.4f}")
    print()
    print(f"information lost by the 224 round trip (A - B): "
          f"{m('ssim_classical') - m('ssim_identity_roundtrip'):+.4f} SSIM")
    print(f"model cost at its own resolution   (C - B): "
          f"{m('ssim_model_at224') - m('ssim_identity_roundtrip'):+.4f} SSIM")
    print(f"cost of the final resize           (D - C): "
          f"{m('ssim_model_roundtrip') - m('ssim_model_at224'):+.4f} SSIM")
    print(f"\nWrote {OUT}")
    return 0





# ---------------------------------------------------------------------------
# Second part: DOMAIN-MATCHED comparison (run separately with --domain-matched)
#
# The proposed model works at 224 px; the baseline U-Net works at the native
# 600 px. Comparing them only after upscaling is unfair to the model. This part
# resizes EVERY system's output (and the reference) to the SAME 224x224 grid and
# recomputes the metrics, so the comparison no longer contains the 2.7x
# upsampling of the model's output.
# ---------------------------------------------------------------------------
def domain_matched() -> int:
    import cv2 as _cv2
    from cnn.dataset import letterbox as _letterbox
    from skimage.metrics import structural_similarity as _ssim, peak_signal_noise_ratio as _psnr
    from src.config import ENHANCED_DIR, RAW_DIR
    from src.preprocess import resize_to_width

    run = "enh224_featguided"
    systems = {
        "raw": lambda n: resize_to_width(_cv2.imread(str(RAW_DIR / n), _cv2.IMREAD_COLOR)),
        "classical": lambda n: _cv2.imread(str(PREPROCESSED_DIR / n), _cv2.IMREAD_COLOR),
        "unet": lambda n: _cv2.imread(str(ENHANCED_DIR / n), _cv2.IMREAD_COLOR),
        "guided": lambda n: _cv2.imread(
            str(ENHANCEMENT_RESULTS_DIR / run / "enhanced" / n), _cv2.IMREAD_COLOR),
    }
    rows = []
    for name in split_ids("test"):
        ref = aligned_reference(name)
        ref224 = _letterbox(_cv2.cvtColor(ref, _cv2.COLOR_BGR2RGB), HYBRID_INPUT_SIZE)
        row = {"image_name": name}
        for key, load in systems.items():
            img = load(name)
            img224 = _letterbox(_cv2.cvtColor(img, _cv2.COLOR_BGR2RGB), HYBRID_INPUT_SIZE)
            h, w = img.shape[:2]
            nh, nw, top, left = letterbox_region(h, w, HYBRID_INPUT_SIZE)
            a = img224[top:top + nh, left:left + nw]
            b = ref224[top:top + nh, left:left + nw]
            row[f"ssim224_{key}"] = float(_ssim(a, b, channel_axis=2, data_range=255))
            row[f"psnr224_{key}"] = float(_psnr(b, a, data_range=255))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "metrics" / "domain_matched_224.csv"
    df.to_csv(out, index=False)
    print("=" * 70)
    print(f"DOMAIN-MATCHED COMPARISON at 224x224 (n={len(df)})")
    print("=" * 70)
    print(f"{'method':<24}{'SSIM@224':>10}{'PSNR@224':>10}")
    for key in systems:
        print(f"{key:<24}{df[f'ssim224_{key}'].mean():>10.4f}{df[f'psnr224_{key}'].mean():>10.4f}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    if "--domain-matched" in sys.argv:
        raise SystemExit(domain_matched())
    raise SystemExit(main())
