"""Torch dataset: preprocessed image + handcrafted features -> (SSIM, PSNR).

- Images: loaded from dataset/preprocessed/, letterboxed (aspect-preserving
  resize + black padding) to CNN_INPUT_SIZE x CNN_INPUT_SIZE, RGB, float32 in
  [0, 1], channel-first. Letterboxing is used instead of stretch-resize so
  texture/edge statistics are not distorted by the CNN input transform.
- Augmentation (TRAIN split only, ``augment=True``): a random element of the
  Klein four-group {identity, hflip, vflip, hflip+vflip} per sample. These four
  were measured to leave all 25 features and both targets exactly invariant, so
  the cached feature/target values stay correct for the augmented image. See
  ``augment_flips`` for why rotations and photometric transforms are excluded.
  Val/test are never augmented: their metric must be a fixed function of the
  split, not of a random draw.
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
    CNN_SEED,
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


def augment_flips(image_rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random element of the Klein four-group {id, hflip, vflip, both}.

    SAFE BY MEASUREMENT, not by assumption. All four transforms were verified
    to leave every one of the 25 handcrafted features unchanged to within
    2.2e-16 relative deviation, and to leave SSIM/PSNR unchanged because the
    identical rigid transform applied to both members of a pair alters neither
    metric. Therefore the cached features and targets remain exactly correct
    for the augmented image and nothing needs recomputing.

    Rotations are deliberately absent: a 90-degree rotation turns the GLCM's
    horizontal adjacency (angles=[0]) into vertical adjacency and shifts the 8
    GLCM descriptors by up to 3.8%, which would silently desynchronise the
    feature branch from the image branch. Photometric transforms are absent
    because they change the targets themselves.
    """
    k = int(rng.integers(0, 4))
    if k == 0:
        return image_rgb
    if k == 1:                                    # horizontal flip
        return np.ascontiguousarray(image_rgb[:, ::-1])
    if k == 2:                                    # vertical flip
        return np.ascontiguousarray(image_rgb[::-1, :])
    return np.ascontiguousarray(image_rgb[::-1, ::-1])   # both



class UIEBQualityDataset(Dataset):
    def __init__(self, split: str, feature_names: list[str],
                 feat_scaler: StandardScaler, target_scaler: StandardScaler,
                 augment: bool = False, seed: int = CNN_SEED):
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split '{split}'")
        df = pd.read_csv(QUALITY_CSV).merge(pd.read_csv(SPLIT_CSV), on="image_name")
        self.rows = df[df["split"] == split].reset_index(drop=True)
        if len(self.rows) == 0:
            raise ValueError(f"no rows for split '{split}'")
        self.feature_names = list(feature_names)
        self.feat_scaler = feat_scaler
        self.target_scaler = target_scaler
        # Augmentation is TRAIN-ONLY. Applying it to val/test would make the
        # reported metric depend on a random draw instead of being a fixed
        # function of the split, and would make runs incomparable.
        self.augment = bool(augment) and split == "train"
        self._rng = np.random.default_rng(seed)
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
        img = letterbox(rgb)
        if self.augment:
            img = augment_flips(img, self._rng)
        img = img.astype(np.float32) / 255.0
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
