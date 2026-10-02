"""Paired dataset for feature-guided enhancement.

WHAT A SAMPLE IS (shapes are printed by the shape test)
-------------------------------------------------------
    image   (3, 224, 224) float32 in [0, 1]   letterboxed PREPROCESSED input
    feats   (k,)          float32             standardised SELECTED features
    target  (3, 224, 224) float32 in [0, 1]   letterboxed aligned REFERENCE
    name    str                               image id (audit trail)

DESIGN DECISIONS (each one is defensible in the viva)
-----------------------------------------------------
1. **Features come from the full-resolution preprocessed image**, exactly as
   the Stage-A statistics used them (cached in feature_quality_dataset.csv).
   They are NOT recomputed from the 224x224 letterbox, because the selected
   features are global statistics and the selection was validated on the
   full-resolution values.  Consistency beats convenience.
2. **The scaler is fitted on TRAIN rows only** (StandardScaler), stored inside
   the checkpoint, and applied unchanged to val/test.  This is the leakage
   guard the audit script re-checks.
3. **The image itself is the letterboxed preprocessed image**, and the target
   is the letterboxed aligned reference (same geometry, both via the frozen
   ``cnn.dataset.letterbox``).  The network learns to map one to the other;
   the un-letterbox + resize back to the preprocessed geometry happens at
   evaluation time (documented resampling, see ``cnn/feature_guided/enhance.py``).
4. **Augmentation is the Klein four-group of flips only**, applied identically
   to input and target (``paired_flips``).  90-degree rotations are EXCLUDED
   because the model consumes cached GLCM-based features: rotating the image
   while keeping the cached feature row would desynchronise the feature branch
   (the GLCM is computed at one angle -- the same reason ``cnn/dataset.py``
   excludes rotations for the quality-prediction models).  A measured residual
   is documented in the docs: under flips, 13 of the 14 selected features are
   identical and ``keypoint_density`` moves by at most ~1%.
5. Val/test are NEVER augmented: their metric must be a fixed function of the
   split, not of a random draw.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from cnn.dataset import letterbox  # noqa: E402  (one letterbox definition)
from cnn.dataset_pairs import load_pair, split_ids, to_tensor  # noqa: E402
from src.config import (  # noqa: E402
    FEATURE_RESULTS_DIR, HYBRID_INPUT_SIZE, HYBRID_SEED, SELECTED_FEATURES_CSV,
)

QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"


def load_selected_features() -> list[str]:
    """The Stage-A selected feature names, in their statistical rank order.

    Reads ``results/statistics/selected_features.csv`` (produced by
    ``scripts/run_statistics.py``).  The count is whatever the funnel produced;
    nothing is hard-coded to 14.
    """
    if not SELECTED_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"{SELECTED_FEATURES_CSV} not found. Run scripts/run_statistics.py "
            f"first (Stage A must exist before Stage B can be trained).")
    df = pd.read_csv(SELECTED_FEATURES_CSV)
    names = df.sort_values("rank")["feature"].astype(str).tolist()
    if len(names) < 2:
        raise ValueError(f"selected feature set too small: {names}")
    return names


def fit_feature_scaler(feature_names: list[str]) -> StandardScaler:
    """StandardScaler fitted on TRAIN rows only (the leakage guard)."""
    df = pd.read_csv(QUALITY_CSV).merge(pd.read_csv(
        FEATURE_RESULTS_DIR / "data_split.csv"), on="image_name")
    train = df[df["split"] == "train"]
    return StandardScaler().fit(train[feature_names].to_numpy(dtype=float))


def paired_flips(img: np.ndarray, tgt: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Apply one of {identity, hflip, vflip, both} IDENTICALLY to both images."""
    k = int(rng.integers(0, 4))
    if k == 0:
        return img, tgt
    out = []
    for a in (img, tgt):
        if k == 1:
            a = a[:, ::-1]
        elif k == 2:
            a = a[::-1, :]
        else:
            a = a[::-1, ::-1]
        out.append(np.ascontiguousarray(a))
    return out[0], out[1]


class HybridPairs(Dataset):
    """(letterboxed input, selected features, letterboxed reference) pairs."""

    def __init__(self, split: str, feature_names: list[str],
                 scaler: StandardScaler, augment: bool = False,
                 size: int = HYBRID_INPUT_SIZE, seed: int = HYBRID_SEED,
                 limit: int | None = None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split '{split}'")
        self.split = split
        self.ids = split_ids(split)
        if limit is not None:                 # smoke/overfit tests only
            self.ids = self.ids[:int(limit)]
        self.feature_names = list(feature_names)
        self.scaler = scaler
        self.size = int(size)
        self.augment = bool(augment) and split == "train"
        self._rng = np.random.default_rng(seed)
        # feature rows for this split, indexed by image_name (audit: the CSV
        # covers all 890 images; we select rows by the frozen split ids)
        self._rows = pd.read_csv(QUALITY_CSV).set_index("image_name")
        missing = [n for n in self.ids if n not in self._rows.index]
        if missing:
            raise ValueError(f"{len(missing)} ids missing from feature CSV, "
                             f"e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        name = self.ids[idx]
        pre_rgb, ref_rgb = load_pair(name)          # full-res RGB uint8, aligned
        img = letterbox(pre_rgb, self.size)
        tgt = letterbox(ref_rgb, self.size)         # identical geometry
        if self.augment:
            img, tgt = paired_flips(img, tgt, self._rng)
        # NOTE: features are the FULL-RESOLUTION statistics of the input image
        # (see module docstring, decision 1); they belong to the same id.
        q = self._rows.loc[name]
        x = self.scaler.transform(
            np.asarray(q[self.feature_names], dtype=float).reshape(1, -1)
        ).astype(np.float32)[0]
        return to_tensor(img), torch.from_numpy(x), to_tensor(tgt), name
