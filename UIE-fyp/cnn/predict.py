"""Single-image inference with a trained hybrid checkpoint.

Given a PREPROCESSED image, extracts the model's handcrafted features from
that same image, letterboxes it for the CNN branch, and predicts SSIM/PSNR
(similarity to the UIEB pseudo-reference the classical pipeline would attain
— see README Limitations).

Usage:
    python -m cnn.predict --run hybrid_final --image dataset/preprocessed/UIEB_0.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn.dataset import letterbox  # noqa: E402
from cnn.model import HybridCNN, ImageOnlyCNN  # noqa: E402
from src.config import MODELS_DIR  # noqa: E402
from src.features import extract_features  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    ckpt = torch.load(MODELS_DIR / f"best_{args.run}.pt", map_location="cpu")
    feat_names = ckpt["feature_names"]
    model = HybridCNN(len(feat_names)) if ckpt["model_name"] == "hybrid" else ImageOnlyCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    feat_scaler = joblib.load(MODELS_DIR / f"feat_scaler_{args.run}.joblib")
    target_scaler = joblib.load(MODELS_DIR / f"target_scaler_{args.run}.joblib")

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"ERROR: cannot read {args.image}")
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = np.transpose(letterbox(rgb).astype(np.float32) / 255.0, (2, 0, 1))
    feats = np.array([extract_features(rgb)[f] for f in feat_names],
                     dtype=float).reshape(1, -1)
    with torch.no_grad():
        pred_n = model(torch.from_numpy(img).unsqueeze(0),
                       torch.from_numpy(feat_scaler.transform(feats).astype(np.float32))
                       ).numpy()
    ssim, psnr = target_scaler.inverse_transform(pred_n)[0]
    print(f"{args.image}: predicted SSIM={ssim:.4f} PSNR={psnr:.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
