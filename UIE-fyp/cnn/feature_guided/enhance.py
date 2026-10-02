"""Run a trained feature-guided enhancer on images -> enhanced PNGs.

RESOLUTION ROUND-TRIP (the part that must be stated honestly)
--------------------------------------------------------------
The model works on a square 224x224 letterboxed view (locked architecture).
To produce an enhanced image comparable to the reference at the preprocessed
geometry, the output is:

    letterboxed 224x224  ->  crop the (nh, nw) region  ->  resize to (H, W)

where (nh, nw) is exactly the region ``cnn.dataset.letterbox`` created out of
the (H, W) preprocessed image.  The final resize is an up-sampling with
INTER_LINEAR.  This resampling is a documented handicap when the same test
set is compared against the full-resolution U-Net baseline; it is disclosed in
``docs/metrics_explanation.md`` and it affects every method here identically
in the proposed-vs-image-only ablation.

FEATURES AT INFERENCE
---------------------
Default: read the cached full-resolution feature row for each image from
``feature_quality_dataset.csv`` (the SAME values training used -- no silent
recomputation drift).  ``--recompute`` extracts the features from the
preprocessed image on the fly, which is what a real deployment would do; the
verification script checks that the two agree exactly.

Usage:
    python -m cnn.feature_guided.enhance --run enh224_featguided --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from cnn.dataset import letterbox  # noqa: E402
from cnn.dataset_pairs import split_ids, to_tensor, to_uint8  # noqa: E402
from cnn.feature_guided.model import FeatureGuidedEnhancer  # noqa: E402
from src.config import (  # noqa: E402
    FEATURE_RESULTS_DIR, HYBRID_INPUT_SIZE, MODELS_DIR, PREPROCESSED_DIR,
)

QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"


def letterbox_region(h: int, w: int, size: int) -> tuple[int, int, int, int]:
    """The (nh, nw, top, left) that ``cnn.dataset.letterbox`` produces.

    Kept as the exact analogue of that function so the round-trip is
    deterministic and auditable.
    """
    scale = size / max(h, w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return nh, nw, (size - nh) // 2, (size - nw) // 2


@torch.no_grad()
def enhance_array(model, bgr_uint8: np.ndarray,
                  feats: np.ndarray | None) -> np.ndarray:
    """One image: BGR uint8 (full resolution) -> enhanced BGR uint8 (same size).

    ``feats`` is the already-standardised feature vector (1, k) float32, or
    None for the image-only variant.
    """
    rgb = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    lb = letterbox(rgb, HYBRID_INPUT_SIZE)                 # 224x224 uint8 RGB
    x = to_tensor(lb).unsqueeze(0)
    was_training = model.training
    model.eval()
    if model.use_features:
        if feats is None:
            raise ValueError("feature-guided model needs the feature vector")
        out = model(x, torch.from_numpy(np.asarray(feats, dtype=np.float32)))
    else:
        out = model(x)
    model.train(was_training)

    nh, nw, top, left = letterbox_region(h, w, HYBRID_INPUT_SIZE)
    crop = out[0][:, top:top + nh, left:left + nw]         # (3, nh, nw) in [0,1]
    img = to_uint8(crop)                                   # RGB uint8 (nh, nw)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def load_model(run_tag: str, device: str = "cpu"):
    """Rebuild the model from ``models/best_<run_tag>.pt`` and load its scaler."""
    ckpt_path = MODELS_DIR / f"best_{run_tag}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FeatureGuidedEnhancer(
        n_features=ckpt["arch"]["n_features"],
        use_features=ckpt["use_features"],
        encoder_widths=tuple(ckpt["arch"]["encoder_widths"]),
        decoder_widths=tuple(ckpt["arch"]["decoder_widths"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def scale_features(ckpt, names: list[str], raw: dict[str, float]) -> np.ndarray:
    """Standardise one feature dict using the checkpoint's train-fitted scaler."""
    x = np.asarray([float(raw[n]) for n in names], dtype=float)
    mean = np.asarray(ckpt["scaler_mean"], dtype=float)
    scale = np.asarray(ckpt["scaler_scale"], dtype=float)
    return ((x - mean) / scale).astype(np.float32).reshape(1, -1)


def enhance_split(run_tag: str, split: str, out_root: Path | None = None,
                  recompute: bool = False, limit: int | None = None) -> Path:
    """Enhance every image of a split and write PNGs + a manifest."""
    model, ckpt = load_model(run_tag)
    names = ckpt["feature_names"]
    ids = split_ids(split)
    if limit:
        ids = ids[:limit]
    out_dir = (out_root or (Path("results") / "enhancement" / run_tag)) / "enhanced"
    out_dir.mkdir(parents=True, exist_ok=True)

    q = pd.read_csv(QUALITY_CSV).set_index("image_name")
    rows = []
    for i, name in enumerate(ids, 1):
        pre_bgr = cv2.imread(str(PREPROCESSED_DIR / name), cv2.IMREAD_COLOR)
        if pre_bgr is None:
            raise FileNotFoundError(f"missing preprocessed image {name}")
        if model.use_features:
            if recompute:
                from src.features import extract_features
                raw = extract_features(cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2RGB))
            else:
                raw = q.loc[name].to_dict()
            feats = scale_features(ckpt, names, raw)
        else:
            feats = None
        enh = enhance_array(model, pre_bgr, feats)
        cv2.imwrite(str(out_dir / name), enh)
        rows.append({"image_name": name, "split": split,
                     "height": int(enh.shape[0]), "width": int(enh.shape[1]),
                     "channels": int(enh.shape[2]),
                     "mean_enh": round(float(enh.astype(np.float64).mean()), 4)})
        if i % 25 == 0 or i == len(ids):
            print(f"  {i}/{len(ids)} enhanced", flush=True)
    pd.DataFrame(rows).to_csv(out_dir.parent / "enhanced_manifest.csv", index=False)
    print(f"Wrote {len(ids)} enhanced images to {out_dir}")
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run tag (checkpoint name)")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--recompute", action="store_true",
                    help="compute features on the fly instead of using the cache")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    enhance_split(args.run, args.split, recompute=args.recompute, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
