"""Training loop for the enhancement / restoration U-Net (image -> image).

Frozen configuration — ``src/config.py``, "Config A" from the measured one-epoch
benchmark. These values were fixed BEFORE this run and are not to be adjusted
against the test split:

    crop 128x128 | batch 8 | base channels 32 -> 472,259 parameters
    Adam lr 2e-4 | L1 pixel-space loss | max 100 epochs | seed 42
    train 623 pairs | validate on 134 | test 133 is NEVER touched here

Protocol
--------
* **Train** on 128x128 random crops with paired geometric augmentation (8
  transforms applied identically to input and target). On-the-fly, so an epoch is
  still 623 samples and costs the benchmarked ~62 s.
* **Validate at FULL resolution** on all 134 val images every ``ENH_VAL_EVERY``
  epochs, reporting mean SSIM, mean PSNR and mean L1. Full resolution matters:
  the metric that will be reported is full-image SSIM, so selecting on a
  128-crop proxy could pick a checkpoint that is good locally and poor globally.
* **Early stopping / model selection on validation SSIM**, patience
  ``ENH_PATIENCE`` validations (= 20 epochs). Best checkpoint is kept, not last.
* **The test split is never loaded.** The train and val id lists are hashed into
  the checkpoint so ``scripts/verify_enhancement.py`` can later prove, from the
  artefacts alone, which images training saw.
* Nothing is written to ``dataset/enhanced-test/``; use ``cnn/enhance.py`` after
  training, then ``scripts/evaluate_enhancement.py``.

Usage:
    python -m cnn.train_enhance
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn.dataset_pairs import (  # noqa: E402
    PairedEnhancementDataset, aligned_reference, enhance_image,
    ids_fingerprint, split_ids,
)
from cnn.unet import EnhancementUNet, count_params  # noqa: E402
from src.config import (  # noqa: E402
    ENH_AUGMENT_GEOMETRIC, ENH_BASE_CHANNELS, ENH_BATCH_SIZE, ENH_CROP_SIZE,
    ENH_EXPECTED_PARAMS, ENH_LOSS, ENH_LR, ENH_MAX_EPOCHS, ENH_PATIENCE,
    ENH_RUN_TAG, ENH_SEED, ENH_VAL_EVERY, ENHANCEMENT_RESULTS_DIR,
    MODELS_DIR, PREPROCESSED_DIR,
)
from src.iqa import compute_ssim_psnr  # noqa: E402


def set_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_full_res(model, val_ids: list[str]) -> dict:
    """Full-resolution validation over the whole val split.

    Returns mean SSIM / PSNR / L1 plus the per-image lists. SSIM and PSNR use
    ``src.iqa.compute_ssim_psnr``, the same definition that produced the labels
    in ``feature_quality_dataset.csv``.
    """
    ssims, psnrs, l1s = [], [], []
    model.eval()
    for name in val_ids:
        pre_bgr = cv2.imread(str(PREPROCESSED_DIR / name), cv2.IMREAD_COLOR)
        ref_bgr = aligned_reference(name)
        enh_bgr = enhance_image(model, pre_bgr)
        s, p = compute_ssim_psnr(enh_bgr, ref_bgr)
        ssims.append(s)
        psnrs.append(p)
        l1s.append(float(np.mean(
            np.abs(enh_bgr.astype(np.float64) - ref_bgr.astype(np.float64))) / 255.0))
    model.train()
    return {"val_ssim": float(np.mean(ssims)), "val_psnr": float(np.mean(psnrs)),
            "val_l1": float(np.mean(l1s)), "val_ssim_list": ssims,
            "val_psnr_list": psnrs}


def train_enhancement(run_tag: str = ENH_RUN_TAG, seed: int = ENH_SEED) -> Path:
    t_start = time.time()
    set_seeds(seed)

    train_ids = split_ids("train")
    val_ids = split_ids("val")          # validation only; test is never loaded
    assert len(train_ids) == 623, f"expected 623 train pairs, got {len(train_ids)}"
    assert len(val_ids) == 134, f"expected 134 val pairs, got {len(val_ids)}"
    assert not (set(train_ids) & set(val_ids)), "train/val overlap"
    test_ids = set(split_ids("test"))
    assert not (set(train_ids) & test_ids), "TEST LEAK into training ids"
    assert not (set(val_ids) & test_ids), "TEST LEAK into validation ids"

    model = EnhancementUNet(ENH_BASE_CHANNELS)
    n_params = count_params(model)
    # Guard the frozen architecture: the benchmark measured exactly this count.
    assert n_params == ENH_EXPECTED_PARAMS, (
        f"architecture drift: {n_params:,} params, frozen config expects "
        f"{ENH_EXPECTED_PARAMS:,}")

    ds = PairedEnhancementDataset("train", crop=ENH_CROP_SIZE, augment=True,
                                  seed=seed)
    assert ds.ids == train_ids, "dataset id order diverged from the split file"
    loader = DataLoader(ds, batch_size=ENH_BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=False,
                        generator=torch.Generator().manual_seed(seed))
    assert len(loader) == 78, f"expected 78 iterations/epoch, got {len(loader)}"

    if ENH_LOSS != "l1":
        raise ValueError(f"unsupported ENH_LOSS {ENH_LOSS!r}; only 'l1' is frozen")
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=ENH_LR,
                                 weight_decay=0.0)

    print(f"Enhancement U-Net | run {run_tag} | seed {seed}")
    print(f"  parameters          : {n_params:,} (frozen = {ENH_EXPECTED_PARAMS:,})")
    print(f"  train pairs         : {len(train_ids)}  (crop {ENH_CROP_SIZE}, "
          f"batch {ENH_BATCH_SIZE}, {len(loader)} iters/epoch)")
    print(f"  val pairs           : {len(val_ids)}  (FULL resolution)")
    print(f"  test ids            : {len(test_ids)} — NOT LOADED")
    print(f"  loss / optimiser    : {ENH_LOSS} / Adam(lr={ENH_LR})")
    print(f"  budget              : max {ENH_MAX_EPOCHS} epochs, validate every "
          f"{ENH_VAL_EVERY}, patience {ENH_PATIENCE} validations on val SSIM")
    print(f"  augmentation        : paired geometric (flips x rot90 = 8) = "
          f"{ENH_AUGMENT_GEOMETRIC}; photometric = False", flush=True)

    out_dir = ENHANCEMENT_RESULTS_DIR / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best = {"ssim": -1.0, "epoch": None, "state": None, "psnr": None, "l1": None}
    bad_validations = 0
    n_validations = 0
    history: list[dict] = []

    for epoch in range(1, ENH_MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, y, _ in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            tot += float(loss.detach()) * len(x)
            n += len(x)
        train_loss = tot / n
        row = {"epoch": epoch, "train_l1": round(train_loss, 6),
               "epoch_seconds": round(time.time() - t0, 1),
               "val_ssim": None, "val_psnr": None, "val_l1": None,
               "is_best": False}

        do_val = (epoch == 1) or (epoch % ENH_VAL_EVERY == 0)
        if do_val:
            tv = time.time()
            v = validate_full_res(model, val_ids)
            n_validations += 1
            row.update({"val_ssim": round(v["val_ssim"], 6),
                        "val_psnr": round(v["val_psnr"], 4),
                        "val_l1": round(v["val_l1"], 6)})
            improved = v["val_ssim"] > best["ssim"] + 1e-6
            if improved:
                best = {"ssim": v["val_ssim"], "epoch": epoch, "psnr": v["val_psnr"],
                        "l1": v["val_l1"],
                        "state": {k: t.cpu().clone()
                                  for k, t in model.state_dict().items()}}
                bad_validations = 0
                row["is_best"] = True
            else:
                bad_validations += 1
            print(f"  epoch {epoch:3d}  train_l1={train_loss:.4f}  "
                  f"val_ssim={v['val_ssim']:.4f}  val_psnr={v['val_psnr']:.3f}  "
                  f"val_l1={v['val_l1']:.4f}  "
                  f"({'*BEST' if improved else f'stale {bad_validations}/{ENH_PATIENCE}'})"
                  f"  [{row['epoch_seconds']:.0f}s train + {time.time()-tv:.0f}s val]",
                  flush=True)
        else:
            if epoch % 5 == 0:
                print(f"  epoch {epoch:3d}  train_l1={train_loss:.4f}  "
                      f"[{row['epoch_seconds']:.0f}s train]", flush=True)

        history.append(row)
        if bad_validations >= ENH_PATIENCE:
            print(f"  early stopping at epoch {epoch}: no val-SSIM improvement "
                  f"for {bad_validations} validations "
                  f"(best {best['ssim']:.4f} @ epoch {best['epoch']})", flush=True)
            break

    assert best["state"] is not None, "no validation ever ran"
    ckpt_path = MODELS_DIR / f"best_{run_tag}.pt"
    torch.save({
        "model_name": "enhancement_unet",
        "run_tag": run_tag,
        "arch": {"base_channels": ENH_BASE_CHANNELS, "levels": EnhancementUNet.LEVELS,
                 "n_params": n_params},
        "config": {"crop": ENH_CROP_SIZE, "batch_size": ENH_BATCH_SIZE,
                   "lr": ENH_LR, "weight_decay": 0.0, "loss": ENH_LOSS,
                   "max_epochs": ENH_MAX_EPOCHS, "val_every": ENH_VAL_EVERY,
                   "patience_validations": ENH_PATIENCE,
                   "augment_geometric": bool(ENH_AUGMENT_GEOMETRIC),
                   "augment_photometric": False},
        "state_dict": best["state"],
        "seed": seed,
        "best_epoch": best["epoch"],
        "best_val_ssim": best["ssim"],
        "best_val_psnr": best["psnr"],
        "best_val_l1": best["l1"],
        "epochs_completed": len(history),
        "n_validations": n_validations,
        # Audit trail: lets verify_enhancement.py PROVE from the artefact alone
        # which images training and validation saw, and that test was excluded.
        "train_ids_fingerprint": ids_fingerprint(train_ids),
        "val_ids_fingerprint": ids_fingerprint(val_ids),
        "n_train": len(train_ids), "n_val": len(val_ids),
        "test_ids_fingerprint": ids_fingerprint(sorted(test_ids)),
        "train_seconds": round(time.time() - t_start, 1),
    }, ckpt_path)
    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    print(f"\nSaved {ckpt_path}")
    print(f"Saved {out_dir / 'train_history.csv'}")
    print(f"  epochs completed {len(history)} | best epoch {best['epoch']} | "
          f"best val SSIM {best['ssim']:.4f} | best val PSNR {best['psnr']:.3f} dB")
    print(f"  total training time {(time.time()-t_start)/60:.1f} min")
    print("\nNEXT: python -m cnn.enhance --split test   (writes enhanced images)")
    print("      python scripts/evaluate_enhancement.py")
    return ckpt_path


def main() -> int:
    train_enhancement()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
