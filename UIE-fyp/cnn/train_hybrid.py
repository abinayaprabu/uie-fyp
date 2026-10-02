"""Train the feature-guided enhancer (or its image-only twin).

Usage
-----
    python -m cnn.train_hybrid --variant feature_guided      # the proposed model
    python -m cnn.train_hybrid --variant image_only          # the ablation twin

Optional flags used by the required pre-flight tests:
    --limit-train 8        train on the first 8 pairs (overfit / smoke test)
    --max-steps 2          cap the optimizer steps per epoch (timing test)
    --epochs 2 --patience 0 --val-every 1 --no-save     quick end-to-end check

Protocol (identical for BOTH variants -- this is what makes the ablation fair)
------------------------------------------------------------------------------
* Train on 623 pairs, validate at FULL RESOLUTION on 134 pairs, never load the
  test split.  Train/val id lists are hashed into the checkpoint.
* Loss: ``HYBRID_LOSS`` from src/config.py (L1 by default; L1 + lambda*(1-SSIM)
  available with an explicit, configurable lambda -- never a value "from a paper").
* Adam, lr 1e-3, weight decay 1e-4, batch 16, early stopping on val SSIM with
  patience 12 epochs, at most 80 epochs.
* The best-by-val-SSIM state is saved, not the last state.
* The feature scaler (train-fitted) is stored inside the checkpoint so the
  evaluation scripts can never silently use different statistics.
* Feature conditioning is identity-initialised: at epoch 0 the feature-guided
  model computes exactly the image-only function.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn.dataset_pairs import aligned_reference, ids_fingerprint, split_ids  # noqa: E402
from cnn.hybrid.dataset import (  # noqa: E402
    HybridPairs, fit_feature_scaler, load_selected_features,
)
from cnn.hybrid.losses import build_loss  # noqa: E402
from cnn.hybrid.model import FeatureGuidedEnhancer, count_params  # noqa: E402
from src.config import (  # noqa: E402
    ENHANCEMENT_RESULTS_DIR, FEATURE_RESULTS_DIR, HYBRID_BATCH_SIZE,
    HYBRID_INPUT_SIZE, HYBRID_LAMBDA_SSIM, HYBRID_LOSS, HYBRID_LR,
    HYBRID_MAX_EPOCHS, HYBRID_PATIENCE, HYBRID_RUN_GUIDED,
    HYBRID_RUN_IMAGE_ONLY, HYBRID_SEED, HYBRID_SSIM_WINDOW, HYBRID_VAL_EVERY,
    HYBRID_WEIGHT_DECAY, MODELS_DIR, PREPROCESSED_DIR,
)
from src.iqa import compute_ssim_psnr  # noqa: E402

QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_threads(n: int | None) -> int:
    """Keep the CPU thread count explicit (2 cores in this sandbox)."""
    if n is None:
        n = min(2, torch.get_num_threads())
    torch.set_num_threads(max(1, int(n)))
    return torch.get_num_threads()


@torch.no_grad()
def validate_full_res(model, val_ids, feature_names, scaler) -> dict:
    """Full-resolution validation: letterbox -> model -> un-letterbox -> metrics.

    The enhanced image is compared against the aligned reference with the
    project's single metric definition (src.iqa). This is the same protocol
    used by the U-Net baseline, so checkpoints are selected the same way.
    """
    import cv2

    from cnn.enhance_hybrid import enhance_array  # local import (no cycle)

    model.eval()
    rows = pd.read_csv(QUALITY_CSV).set_index("image_name")
    ssims, psnrs, l1s = [], [], []
    for name in val_ids:
        pre_bgr = cv2.imread(str(PREPROCESSED_DIR / name), cv2.IMREAD_COLOR)
        ref_bgr = aligned_reference(name)
        feats = None
        if model.use_features:
            x = scaler.transform(
                np.asarray(rows.loc[name, feature_names], dtype=float)
                .reshape(1, -1)).astype(np.float32)
            feats = x
        enh_bgr = enhance_array(model, pre_bgr, feats)
        s, p = compute_ssim_psnr(enh_bgr, ref_bgr)
        ssims.append(s)
        psnrs.append(p)
        l1s.append(float(np.mean(np.abs(enh_bgr.astype(np.float64)
                                       - ref_bgr.astype(np.float64))) / 255.0))
    # restore train mode only if the caller was training
    model.train()
    return {"val_ssim": float(np.mean(ssims)), "val_psnr": float(np.mean(psnrs)),
            "val_l1": float(np.mean(l1s))}


def build_and_train(variant: str, run_tag: str, seed: int, epochs: int | None,
                    patience: int | None, batch: int, lr: float, val_every: int,
                    limit_train: int | None, max_steps: int | None,
                    device_name: str, save: bool, threads: int | None,
                    loss_name: str | None, lam: float | None) -> Path | None:
    set_seeds(seed)
    n_threads = set_threads(threads)
    device = torch.device(device_name)

    feature_names = load_selected_features()
    use_features = variant == "feature_guided"
    scaler = fit_feature_scaler(feature_names)

    train_ids = split_ids("train")
    val_ids = split_ids("val")
    test_ids = split_ids("test")
    assert len(train_ids) == 623 and len(val_ids) == 134 and len(test_ids) == 133
    assert not (set(train_ids) & set(val_ids))
    assert not (set(train_ids) & set(test_ids))
    assert not (set(val_ids) & set(test_ids))

    ds_tr = HybridPairs("train", feature_names, scaler, augment=True, seed=seed,
                        limit=limit_train)
    ds_va = HybridPairs("val", feature_names, scaler, augment=False, seed=seed)
    loader = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=0,
                        drop_last=False,
                        generator=torch.Generator().manual_seed(seed))

    model = FeatureGuidedEnhancer(
        n_features=len(feature_names), use_features=use_features).to(device)
    n_params = count_params(model)

    criterion = build_loss(loss_name or HYBRID_LOSS,
                           lam=HYBRID_LAMBDA_SSIM if lam is None else lam,
                           window=HYBRID_SSIM_WINDOW)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=HYBRID_WEIGHT_DECAY)

    print(f"Variant            : {variant}   (use_features={use_features})")
    print(f"Run tag            : {run_tag}")
    print(f"Selected features  : {len(feature_names)} -> {feature_names}")
    print(f"Parameters         : {n_params:,}")
    print(f"Device / threads   : {device} / {n_threads}")
    print(f"Loss               : {criterion.__class__.__name__} "
          f"(name={loss_name or HYBRID_LOSS}, lambda={lam if lam is not None else HYBRID_LAMBDA_SSIM})")
    print(f"Train pairs        : {len(ds_tr)} (batch {batch}, {len(loader)} iters)")
    print(f"Val pairs          : {len(ds_va)} (full resolution, every {val_every} epochs)")
    print(f"Test ids           : {len(test_ids)} -- NOT LOADED")
    if limit_train:
        print("  ** SMOKE / OVERFIT MODE: not a final training run **")
    if max_steps:
        print("  ** per-epoch step cap active: not a final training run **")
    print(f"Identifiers        : train {ids_fingerprint(train_ids)} | "
          f"val {ids_fingerprint(val_ids)} | test {ids_fingerprint(test_ids)}",
          flush=True)

    out_dir = ENHANCEMENT_RESULTS_DIR / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    best = {"ssim": -1.0, "epoch": None, "state": None, "psnr": None, "l1": None}
    bad, history = 0, []
    epoch_budget = epochs if epochs is not None else HYBRID_MAX_EPOCHS
    patience_eff = HYBRID_PATIENCE if patience is None else patience

    for epoch in range(1, epoch_budget + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for step, (img, feat, tgt, _) in enumerate(loader):
            if max_steps and step >= max_steps:
                break
            img, tgt = img.to(device), tgt.to(device)
            feat = feat.to(device)
            optimizer.zero_grad()
            out = model(img, feat if use_features else None)
            loss = criterion(out, tgt)
            loss.backward()
            optimizer.step()
            tot += float(loss.detach()) * len(img)
            n += len(img)
        train_loss = tot / max(n, 1)
        row = {"epoch": epoch, "train_loss": round(train_loss, 6),
               "epoch_seconds": round(time.time() - t0, 1),
               "val_ssim": None, "val_psnr": None, "val_l1": None,
               "is_best": False}

        if val_every and epoch % val_every == 0:
            tv = time.time()
            v = validate_full_res(model, val_ids, feature_names, scaler)
            row.update({"val_ssim": round(v["val_ssim"], 6),
                        "val_psnr": round(v["val_psnr"], 4),
                        "val_l1": round(v["val_l1"], 6)})
            improved = v["val_ssim"] > best["ssim"] + 1e-6
            if improved:
                best = {"ssim": v["val_ssim"], "epoch": epoch,
                        "psnr": v["val_psnr"], "l1": v["val_l1"],
                        "state": {k: t.cpu().clone()
                                  for k, t in model.state_dict().items()}}
                bad = 0
                row["is_best"] = True
            else:
                bad += 1
            print(f"  epoch {epoch:3d}  train={train_loss:.5f}  "
                  f"val_ssim={v['val_ssim']:.4f}  val_psnr={v['val_psnr']:.3f}  "
                  f"({'*BEST' if improved else f'stale {bad}/{patience_eff}'})  "
                  f"[{row['epoch_seconds']:.0f}s + {time.time()-tv:.0f}s val]",
                  flush=True)
            if patience_eff > 0 and bad >= patience_eff:
                print(f"  early stop at epoch {epoch} (best val SSIM "
                      f"{best['ssim']:.4f} @ epoch {best['epoch']})", flush=True)
                history.append(row)
                break

        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)

    ckpt_path = None
    if save and best["state"] is not None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = MODELS_DIR / f"best_{run_tag}.pt"
        torch.save({
            "model_name": "feature_guided_enhancer",
            "variant": variant,
            "use_features": use_features,
            "run_tag": run_tag,
            "arch": {
                "n_features": len(feature_names),
                "encoder_widths": [32, 64, 128, 256],
                "decoder_widths": [128, 64, 32, 16],
                "input_size": HYBRID_INPUT_SIZE,
                "n_params": n_params,
            },
            "feature_names": feature_names,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "state_dict": best["state"],
            "seed": seed,
            "best_epoch": best["epoch"],
            "best_val_ssim": best["ssim"],
            "best_val_psnr": best["psnr"],
            "loss": {"name": loss_name or HYBRID_LOSS,
                     "lambda_ssim": lam if lam is not None else HYBRID_LAMBDA_SSIM},
            "train_ids_fingerprint": ids_fingerprint(train_ids),
            "val_ids_fingerprint": ids_fingerprint(val_ids),
            "test_ids_fingerprint": ids_fingerprint(test_ids),
            "n_train": len(ds_tr), "n_val": len(val_ids),
            "epochs_completed": len(history),
        }, ckpt_path)
        print(f"Saved {ckpt_path}")
        # Durability copy: the sandbox has destroyed gitignored files before.
        # results/ is tracked, so keep a byte-identical copy of the checkpoint
        # next to the run's metrics (the .gitignore explicitly allows this one
        # path; ~5 MB per run).
        import shutil
        durable = out_dir / f"best_{run_tag}.pt"
        shutil.copyfile(ckpt_path, durable)
        print(f"Saved durable copy {durable}")
    print(f"History -> {out_dir / 'train_history.csv'}")
    if best["state"] is not None:
        print(f"Best val SSIM {best['ssim']:.4f} @ epoch {best['epoch']} "
              f"(val PSNR {best['psnr']:.3f} dB)")
    return ckpt_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=("image_only", "feature_guided"),
                    default="feature_guided")
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--seed", type=int, default=HYBRID_SEED)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch", type=int, default=HYBRID_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=HYBRID_LR)
    ap.add_argument("--val-every", type=int, default=HYBRID_VAL_EVERY)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--loss", default=None)
    ap.add_argument("--lambda-ssim", type=float, default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    run_tag = args.run_tag or (HYBRID_RUN_GUIDED if args.variant == "feature_guided"
                               else HYBRID_RUN_IMAGE_ONLY)
    build_and_train(args.variant, run_tag, args.seed, args.epochs, args.patience,
                    args.batch, args.lr, args.val_every, args.limit_train,
                    args.max_steps, args.device, not args.no_save, args.threads,
                    args.loss, args.lambda_ssim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
