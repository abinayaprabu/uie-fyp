"""Paired image-to-image dataset for the enhancement U-Net.

This is the counterpart of ``cnn/dataset.py`` for the *enhancement* half of the
framework. The difference is fundamental:

    cnn/dataset.py        (image, 14 features) -> 2 scalars  [SSIM, PSNR]
    cnn/dataset_pairs.py  (image)              -> image      [the reference]

Inputs and guarantees
---------------------
* **Input**  = ``dataset/preprocessed/<id>.png`` — the frozen classical
  pipeline's output. This module never re-implements or alters preprocessing.
* **Target** = ``dataset/reference-890/<id>.png`` passed through the project's
  OWN frozen ``src.preprocess.resize_reference_like_preprocessed``, which is the
  identical geometric resize the raw image underwent. This is what makes the
  pair pixel-aligned, and it is exactly how the SSIM/PSNR *labels* in
  ``feature_quality_dataset.csv`` were computed, so the enhancement target and
  the quality-assessment target are the same object.
* **Split** = ``results/feature/data_split.csv``, the frozen group-aware
  623/134/133 split. Constructor asserts the requested split is non-empty and
  records the ids so a later audit can prove the test split was never trained on.
* **Reference images are targets, never inputs** — the network sees only the
  preprocessed image.

Augmentation
------------
Paired **geometric** transforms only: ``{id, hflip, vflip, both} x {rot0, rot90,
rot180, rot270}`` = 8, applied *identically* to input and target. A rigid
transform applied to both members of a pair leaves the pair consistent, so the
supervision stays exactly correct and the effective dataset is 8x larger at zero
extra I/O (it is applied on the fly, so an epoch is still 623 samples).

90-degree rotations are **safe here but were banned in ``cnn/dataset.py``**. The
reason is not caution, it is measurement: rot90 turns the GLCM's horizontal
adjacency (``angles=[0]``) into vertical adjacency and shifts the 8 handcrafted
GLCM descriptors by up to 3.8%, which would desynchronise the feature branch.
This dataset has no handcrafted features, so that constraint does not apply.

**Photometric augmentation is excluded outright.** Brightness/contrast/colour/
gamma jitter would change the input *without* changing the target, i.e. it would
teach the network that a wrongly-coloured image should map to the untouched
reference. That is not regularisation, it is label noise.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    ENH_AUGMENT_GEOMETRIC, ENH_CROP_SIZE, ENH_SEED, PREPROCESSED_DIR,
    REFERENCE_DIR, SPLIT_CSV,
)
from src.preprocess import resize_reference_like_preprocessed  # noqa: E402


# --------------------------------------------------------------------------
# alignment / conversion helpers
# --------------------------------------------------------------------------
def aligned_reference(name: str) -> np.ndarray:
    """The UIEB reference for ``name``, geometrically aligned to the
    preprocessed image using the project's frozen resize. BGR uint8."""
    ref = cv2.imread(str(REFERENCE_DIR / name), cv2.IMREAD_COLOR)
    if ref is None:
        raise ValueError(f"unreadable reference image {name}")
    return resize_reference_like_preprocessed(ref)


def load_pair(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(preprocessed, aligned reference) as RGB uint8, shapes asserted equal."""
    pre = cv2.imread(str(PREPROCESSED_DIR / name), cv2.IMREAD_COLOR)
    if pre is None:
        raise ValueError(f"unreadable preprocessed image {name}")
    ref = aligned_reference(name)
    if pre.shape != ref.shape:
        raise ValueError(f"pair {name} not pixel-aligned: preprocessed "
                         f"{pre.shape} vs reference {ref.shape}")
    return (cv2.cvtColor(pre, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))


def to_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    """RGB uint8 HWC -> float32 CHW in [0, 1]."""
    return torch.from_numpy(
        np.transpose(rgb_uint8.astype(np.float32) / 255.0, (2, 0, 1)))


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """float CHW/1 CHW in [0,1] -> RGB uint8 HWC. Round-half-up, then clip.

    Quantisation to uint8 is deliberate and documented: the project's SSIM/PSNR
    definition (``scripts/build_feature_dataset.py::compute_ssim_psnr``) is
    defined on uint8 images with ``data_range=255``, so the enhanced output must
    be quantised identically for the comparison to be apples-to-apples.
    """
    a = np.asarray(t.detach().cpu(), dtype=np.float64)
    if a.ndim == 4:
        a = a[0]
    a = np.transpose(a, (1, 2, 0))
    return np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)


def pad_to_multiple(x: torch.Tensor, m: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Reflect-pad (..., H, W) so H and W are divisible by ``m``.

    The U-Net pools twice, so full-resolution inference needs H, W % 4 == 0.
    UIEB heights run 266-901 at a fixed width of 600, so padding is required.
    Returns the padded tensor and the ORIGINAL (H, W) for cropping back.
    """
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (h, w)


def enhance_image(model, bgr_uint8: np.ndarray) -> np.ndarray:
    """Run the frozen U-Net on one full-resolution image. BGR uint8 in/out.

    Batch 1 under ``no_grad``. Measured worst case (875x600, the largest UIEB
    validation image): 1.19 s and +606 MB peak RSS on 2 CPU cores.
    """
    rgb = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB)
    x = to_tensor(rgb).unsqueeze(0)
    x, (h, w) = pad_to_multiple(x, 4)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(x)
    model.train(was_training)
    out = out[0][:, :h, :w]
    return cv2.cvtColor(to_uint8(out), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# paired geometric augmentation
# --------------------------------------------------------------------------
def paired_geometric(img: np.ndarray, tgt: np.ndarray,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Apply one of 8 rigid transforms IDENTICALLY to input and target."""
    k = int(rng.integers(0, 4))            # rot90 multiplier
    fh = bool(rng.integers(0, 2))          # horizontal flip
    fv = bool(rng.integers(0, 2))          # vertical flip
    out = []
    for a in (img, tgt):
        if k:
            a = np.rot90(a, k, axes=(0, 1))
        if fh:
            a = a[:, ::-1]
        if fv:
            a = a[::-1, :]
        out.append(np.ascontiguousarray(a))
    return out[0], out[1]


# --------------------------------------------------------------------------
def split_ids(split: str) -> list[str]:
    """The frozen image ids for one split, in file order."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split '{split}'")
    df = pd.read_csv(SPLIT_CSV)
    ids = df.loc[df["split"] == split, "image_name"].tolist()
    if not ids:
        raise ValueError(f"no rows for split '{split}'")
    return ids


def ids_fingerprint(ids: list[str]) -> str:
    """Stable hash of an id list, recorded in the checkpoint so an auditor can
    later PROVE which images training actually saw."""
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


class PairedEnhancementDataset(Dataset):
    """Train/val pairs. Training uses random crops + augmentation; the
    validation pass in ``cnn/train_enhance.py`` runs full-resolution instead,
    so ``crop=None`` returns whole images."""

    def __init__(self, split: str, crop: int | None = ENH_CROP_SIZE,
                 augment: bool | None = None, seed: int = ENH_SEED):
        self.split = split
        self.ids = split_ids(split)
        self.crop = crop
        if augment is None:
            augment = ENH_AUGMENT_GEOMETRIC
        # Augmentation is TRAIN-ONLY. Augmenting val would make the reported
        # validation metric depend on a random draw instead of being a fixed
        # function of the split, which would make early stopping noisy.
        self.augment = bool(augment) and split == "train"
        self._rng = np.random.default_rng(seed)
        self.fingerprint = ids_fingerprint(self.ids)
        missing = [n for n in self.ids
                   if not (PREPROCESSED_DIR / n).exists()
                   or not (REFERENCE_DIR / n).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} pairs missing on disk, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        name = self.ids[idx]
        pre, ref = load_pair(name)
        if self.augment:
            pre, ref = paired_geometric(pre, ref, self._rng)
        if self.crop is not None:
            h, w = pre.shape[:2]
            c = self.crop
            if h < c or w < c:
                raise ValueError(f"{name} is {h}x{w}, smaller than crop {c}")
            y = int(self._rng.integers(0, h - c + 1))
            x = int(self._rng.integers(0, w - c + 1))
            pre = pre[y:y + c, x:x + c]
            ref = ref[y:y + c, x:x + c]
        return to_tensor(pre), to_tensor(ref), name
