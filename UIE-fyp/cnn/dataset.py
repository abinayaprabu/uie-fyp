"""Torch dataset: preprocessed image + handcrafted features -> (SSIM, PSNR).

- Images: loaded from dataset/preprocessed/, letterboxed (aspect-preserving
  resize + black padding) to CNN_INPUT_SIZE x CNN_INPUT_SIZE, RGB, float32 in
  [0, 1], channel-first. Letterboxing is used instead of stretch-resize so
  texture/edge statistics are not distorted by the CNN input transform.
- Handcrafted features: from feature_quality_dataset.csv, standardised with a
  StandardScaler FIT ON TRAIN ONLY (CHECK 12); the fitted scaler is saved.
- Targets: SSIM/PSNR standardised with a second StandardScaler FIT ON TRAIN
  ONLY (the two targets live on very different scales: SSIM ~[0,1] vs PSNR
  ~10-40 dB; normalising lets one MSE loss treat both outputs fairly).
  Predictions are inverse-transformed before metric reporting.
- Rows are joined to data_split.csv by image_name (CHECK 14); reference
  images are NEVER model inputs (they only generated the target columns).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    CNN_INPUT_SIZE,
    FEATURE_RESULTS_DIR,
    PREPROCESSED_DIR,
    SPLIT_CSV,
)

QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"


def letterbox(image_rgb: np.ndarray, size: int = CNN_INPUT_SIZE) -> np.ndarray:
    """Aspect-preserving resize + black padding to (size, size)."""
    h, w = image_rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


class UIEBQualityDataset(Dataset):
    def __init__(self, split: str, feature_names: list[str],
                 feat_scaler: StandardScaler, target_scaler: StandardScaler):
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split '{split}'")
        df = pd.read_csv(QUALITY_CSV).merge(pd.read_csv(SPLIT_CSV), on="image_name")
        self.rows = df[df["split"] == split].reset_index(drop=True)
        if len(self.rows) == 0:
            raise ValueError(f"no rows for split '{split}'")
        self.feature_names = list(feature_names)
        self.feat_scaler = feat_scaler
        self.target_scaler = target_scaler
        # CHECK 14 evidence: every feature row must have its image on disk.
        missing = [n for n in self.rows["image_name"]
                   if not (PREPROCESSED_DIR / n).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} preprocessed images missing, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        bgr = cv2.imread(str(PREPROCESSED_DIR / row["image_name"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"unreadable image {row['image_name']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = letterbox(rgb).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        feats = self.feat_scaler.transform(
            row[self.feature_names].to_numpy(dtype=float).reshape(1, -1)
        ).astype(np.float32)[0]
        target = self.target_scaler.transform(
            row[["ssim", "psnr"]].to_numpy(dtype=float).reshape(1, -1)
        ).astype(np.float32)[0]
        return (torch.from_numpy(img), torch.from_numpy(feats),
                torch.from_numpy(target), row["image_name"])


def fit_scalers(feature_names: list[str]) -> tuple[StandardScaler, StandardScaler]:
    """Fit feature + target scalers on TRAIN rows only (CHECK 12)."""
    df = pd.read_csv(QUALITY_CSV).merge(pd.read_csv(SPLIT_CSV), on="image_name")
    train = df[df["split"] == "train"]
    feat_scaler = StandardScaler().fit(train[feature_names].to_numpy(dtype=float))
    target_scaler = StandardScaler().fit(train[["ssim", "psnr"]].to_numpy(dtype=float))
    return feat_scaler, target_scaler
