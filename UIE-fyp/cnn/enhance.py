"""Run the trained enhancement U-Net and WRITE the enhanced images to disk.

This is the inference half of the enhancement component: the missing piece the
audit identified was not just a network but a network whose output is *saved as
an image*. Everything downstream (``scripts/evaluate_enhancement.py`` and
``scripts/verify_enhancement.py``) reads those PNGs back from disk, so the
reported metrics are measured on the artefacts a human can look at — not on
tensors that existed only in memory during training.

Full-resolution inference: the U-Net pools twice, so the input is reflect-padded
to a multiple of 4 and cropped back afterwards (``cnn.dataset_pairs``). Measured
worst case on this machine (875x600, the largest UIEB image): 1.19 s and +606 MB.

Usage:
    python -m cnn.enhance --split test            # the sealed 133 test images
    python -m cnn.enhance --split val             # for inspection only
    python -m cnn.enhance --split test --limit 5  # quick smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn.dataset_pairs import enhance_image, split_ids  # noqa: E402
from cnn.unet import EnhancementUNet  # noqa: E402
from src.config import (  # noqa: E402
    ENH_RUN_TAG, ENHANCEMENT_RESULTS_DIR, ENHANCED_DIR, MODELS_DIR,
    PREPROCESSED_DIR,
)


def load_model(run_tag: str = ENH_RUN_TAG):
    path = MODELS_DIR / f"best_{run_tag}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `python -m cnn.train_enhance` first.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("model_name") != "enhancement_unet":
        raise ValueError(f"{path} is not an enhancement checkpoint "
                         f"(model_name={ckpt.get('model_name')!r})")
    model = EnhancementUNet(ckpt["arch"]["base_channels"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=ENH_RUN_TAG)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N images (smoke test)")
    ap.add_argument("--out", default=None,
                    help=f"output dir (default {ENHANCED_DIR})")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else ENHANCED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model(args.run)
    ids = split_ids(args.split)
    if args.limit:
        ids = ids[:args.limit]

    print(f"Enhancing {len(ids)} {args.split}-split images with "
          f"{args.run} (best epoch {ckpt['best_epoch']}, "
          f"val SSIM {ckpt['best_val_ssim']:.4f})")
    print(f"  output -> {out_dir}")
    t0 = time.time()
    rows = []
    for i, name in enumerate(ids, 1):
        pre = cv2.imread(str(PREPROCESSED_DIR / name), cv2.IMREAD_COLOR)
        if pre is None:
            raise ValueError(f"unreadable preprocessed image {name}")
        enh = enhance_image(model, pre)
        if enh.shape != pre.shape:
            raise ValueError(f"{name}: shape changed {pre.shape} -> {enh.shape}")
        if enh.dtype != np.uint8:
            raise ValueError(f"{name}: output is {enh.dtype}, expected uint8")
        cv2.imwrite(str(out_dir / name), enh)
        rows.append({"image_name": name, "split": args.split,
                     "height": enh.shape[0], "width": enh.shape[1],
                     "channels": enh.shape[2],
                     "mean_pre": round(float(pre.mean()), 4),
                     "mean_enh": round(float(enh.mean()), 4)})
        if i % 25 == 0 or i == len(ids):
            print(f"  {i}/{len(ids)} ({(time.time()-t0)/i:.2f} s/image)", flush=True)

    manifest = ENHANCEMENT_RESULTS_DIR / args.run / f"enhanced_{args.split}_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"\nWrote {len(rows)} enhanced images + {manifest}")
    print(f"  elapsed {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
