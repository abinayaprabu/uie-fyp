#!/usr/bin/env python3
"""Run the classical enhancement pipeline: raw-890 -> preprocessed/.

Reads every raw image with OpenCV (BGR), applies src.preprocess.enhance()
(resize width 600 -> gray-world -> CLAHE(L) -> bilateral -> adaptive gamma)
and writes the result as PNG with the SAME filename into dataset/preprocessed/.

Also writes results/feature/preprocessing_log.csv (image_name, gamma, out
shape) so the adaptive-gamma behaviour is auditable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import FEATURE_RESULTS_DIR, PREPROCESSED_DIR, RAW_DIR  # noqa: E402
from src.preprocess import enhance  # noqa: E402


def main() -> int:
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in RAW_DIR.glob("*") if p.is_file())
    if not files:
        print(f"ERROR: no raw images found in {RAW_DIR}. "
              "Run scripts/download_uieb.py first.", file=sys.stderr)
        return 1
    print(f"Preprocessing {len(files)} raw images -> {PREPROCESSED_DIR}")
    log = []
    for path in tqdm(files, desc="preprocess", unit="img"):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"ERROR: unreadable image {path}", file=sys.stderr)
            return 1
        out, info = enhance(img)
        cv2.imwrite(str(PREPROCESSED_DIR / path.name), out)
        log.append({"image_name": path.name, "gamma": round(info["gamma"], 4),
                    "out_h": out.shape[0], "out_w": out.shape[1]})
    df = pd.DataFrame(log)
    df.to_csv(FEATURE_RESULTS_DIR / "preprocessing_log.csv", index=False)
    print(f"Done. gamma: min={df['gamma'].min():.3f} mean={df['gamma'].mean():.3f} "
          f"max={df['gamma'].max():.3f}")
    print(f"Output widths unique: {sorted(df['out_w'].unique())} "
          f"(all 600 expected); heights: min={df['out_h'].min()} max={df['out_h'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
