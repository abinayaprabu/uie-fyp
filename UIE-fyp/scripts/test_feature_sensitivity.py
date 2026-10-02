#!/usr/bin/env python3
"""Inference-time feature sensitivity of the trained feature-guided model.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It is NOT the required trained ablation (image-only vs feature-guided), which
needs a second training run. It is a cheap sensitivity test of the ALREADY
TRAINED model: the same test images are enhanced three times while only the
feature vector changes (normal / zeroed / shuffled), and the metrics are
recomputed from the resulting images. It answers "how much does this trained
model actually use the handcrafted guidance?" -- not "would the model be worse
without it during training?".

If zeroing or shuffling barely moves the metrics, that is evidence the model
learned to rely mostly on the image branch, and no claim about feature guidance
can be made from this model alone.

Writes results/metrics/feature_sensitivity.csv
Usage:  python scripts/test_feature_sensitivity.py --run enh224_featguided
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
    FEATURE_RESULTS_DIR, HYBRID_INPUT_SIZE, PREPROCESSED_DIR, RESULTS_DIR,
)
from src.iqa import compute_ssim_psnr  # noqa: E402

QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"


@torch.no_grad()
def enhance(model, pre_bgr, feats):
    rgb = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    lb = letterbox(rgb, HYBRID_INPUT_SIZE)
    x = to_tensor(lb).unsqueeze(0)
    out = model(x, feats)[0]
    nh, nw, top, left = letterbox_region(h, w, HYBRID_INPUT_SIZE)
    crop = to_uint8(out[:, top:top + nh, left:left + nw])
    img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="enh224_featguided")
    args = ap.parse_args()

    model, ckpt = load_model(args.run)
    names = ckpt["feature_names"]
    q = pd.read_csv(QUALITY_CSV).set_index("image_name")
    rng = np.random.default_rng(42)

    rows = []
    for name in split_ids("test"):
        ref = aligned_reference(name)
        pre = cv2.imread(str(PREPROCESSED_DIR / name), cv2.IMREAD_COLOR)
        f = torch.from_numpy(scale_features(ckpt, names, q.loc[name].to_dict()))
        variants = {
            "normal": f,
            "zeroed": torch.zeros_like(f),
            "shuffled": f[:, torch.from_numpy(rng.permutation(len(names)))],
        }
        row = {"image_name": name}
        for tag, feat in variants.items():
            img = enhance(model, pre, feat)
            s, p = compute_ssim_psnr(img, ref)
            row[f"ssim_{tag}"] = s
            row[f"psnr_{tag}"] = p
        rows.append(row)

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "metrics" / "feature_sensitivity.csv"
    df.to_csv(out, index=False)

    print("=" * 74)
    print(f"INFERENCE-TIME FEATURE SENSITIVITY (n={len(df)}, run {args.run})")
    print("=" * 74)
    print("NOT a trained ablation -- the model is unchanged, only its input "
          "features change.\n")
    print(f"{'features':<12}{'SSIM':>10}{'PSNR':>10}")
    for tag in ("normal", "zeroed", "shuffled"):
        print(f"{tag:<12}{df[f'ssim_{tag}'].mean():>10.4f}{df[f'psnr_{tag}'].mean():>10.4f}")
    d_s = (df["ssim_normal"] - df["ssim_zeroed"]).mean()
    d_p = (df["psnr_normal"] - df["psnr_zeroed"]).mean()
    print(f"\nnormal - zeroed : dSSIM {d_s:+.4f}  dPSNR {d_p:+.4f} dB")
    d_s2 = (df["ssim_normal"] - df["ssim_shuffled"]).mean()
    d_p2 = (df["psnr_normal"] - df["psnr_shuffled"]).mean()
    print(f"normal - shuffled: dSSIM {d_s2:+.4f}  dPSNR {d_p2:+.4f} dB")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
