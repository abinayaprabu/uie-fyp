"""Training loop shared by all neural models (MLP / image-only / hybrid).

Protocol (identical for every neural model so comparisons are fair):
- Adam(lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY), MSE loss on NORMALISED
  targets (equal weighting is justified because each target was standardised
  with train statistics, so both have unit variance at training time).
- Early stopping on val loss (patience=CNN_PATIENCE), best checkpoint kept.
- Fixed seeds (random/numpy/torch, single-threaded deterministic data order
  via generator seed) for reproducibility.
- Saves: best_model.pt, train_history.csv, scalers (joblib), feature list.

Usage (from project root):
    python -m cnn.train --model hybrid --features final   # hybrid + final feats
    python -m cnn.train --model hybrid --features all25   # ablation C
    python -m cnn.train --model image_only                # baseline 3 / ablation A
    python -m cnn.train --model mlp --features final      # baseline 2 (MLP)
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn.dataset import UIEBQualityDataset, fit_scalers  # noqa: E402
from cnn.model import FeatMLP, HybridCNN, ImageOnlyCNN, count_params  # noqa: E402
from src.config import (  # noqa: E402
    CNN_AUGMENT_FLIPS,
    CNN_BATCH_SIZE,
    CNN_LR,
    CNN_MAX_EPOCHS,
    CNN_PATIENCE,
    CNN_RESULTS_DIR,
    CNN_SEED,
    CNN_WEIGHT_DECAY,
    FEATURE_NAMES_25,
    FEATURE_RESULTS_DIR,
    MODELS_DIR,
)


def set_seeds(seed: int = CNN_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # NOTE: an earlier version called torch.set_num_threads(max(1, torch.get_num_threads()))
    # here, which is a no-op (it sets the thread count to itself). Removed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_features(which: str) -> list[str]:
    if which == "all25":
        return list(FEATURE_NAMES_25)
    if which == "final":
        path = FEATURE_RESULTS_DIR / "final_selected_features.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run scripts/feature_subset_evaluation.py first.")
        return (pd.read_csv(path).sort_values("rank")["feature"].tolist())
    raise ValueError(f"unknown --features '{which}' (use 'final' or 'all25')")


def build_model(name: str, n_features: int):
    if name == "hybrid":
        return HybridCNN(n_features)
    if name == "image_only":
        return ImageOnlyCNN()
    if name == "mlp":
        return FeatMLP(n_features)
    raise ValueError(f"unknown --model '{name}'")


def run_epoch(model, loader, criterion, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    for images, feats, targets, _ in loader:
        if train:
            optimizer.zero_grad()
        preds = model(images, feats)
        loss = criterion(preds, targets)
        if train:
            loss.backward()
            optimizer.step()
        total += loss.item() * len(images)
        n += len(images)
    return total / n


def train_model(model_name: str, features_which: str, seed: int = CNN_SEED,
                augment: bool | None = None) -> Path:
    """Train one model. ``seed`` varies initialisation AND batch order.

    ``augment=None`` means "use CNN_AUGMENT_FLIPS from config". Augmentation is
    applied to the train split only (enforced inside UIEBQualityDataset).

    The default seed keeps the historic run tag (e.g. ``hybrid_final``) so that
    run_baselines.py / run_ablation.py keep working unchanged; other seeds get
    an explicit ``_s<seed>`` suffix so repeated runs never overwrite each other.
    """
    t0 = time.time()
    if augment is None:
        augment = CNN_AUGMENT_FLIPS
    set_seeds(seed)
    feat_names = [] if model_name == "image_only" else resolve_features(features_which)
    n_features = len(feat_names) if feat_names else 1  # dummy width if unused
    feat_scaler, target_scaler = (
        fit_scalers(feat_names) if feat_names
        else fit_scalers([FEATURE_NAMES_25[0]])  # unused dummy; keeps API uniform
    )
    run_tag = f"{model_name}_{features_which if feat_names else 'nofeat'}"
    if seed != CNN_SEED:
        run_tag += f"_s{seed}"
    run_dir = CNN_RESULTS_DIR / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        UIEBQualityDataset("train", feat_names or [FEATURE_NAMES_25[0]],
                           feat_scaler, target_scaler, augment=augment, seed=seed),
        batch_size=CNN_BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(
        UIEBQualityDataset("val", feat_names or [FEATURE_NAMES_25[0]],
                           feat_scaler, target_scaler, augment=False),
        batch_size=CNN_BATCH_SIZE, shuffle=False)

    model = build_model(model_name, n_features)
    print(f"Model {model_name} ({count_params(model):,} params) | seed {seed} | "
          f"augment(flips)={augment} | features: "
          f"{feat_names if feat_names else 'none (image only)'}")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LR,
                                 weight_decay=CNN_WEIGHT_DECAY)

    best_val, best_state, bad = float("inf"), None, 0
    history = []
    for epoch in range(1, CNN_MAX_EPOCHS + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer)
        va = run_epoch(model, val_loader, criterion, None)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        improved = va < best_val - 1e-6
        if improved:
            best_val, bad = va, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch == 1 or epoch % 5 == 0 or improved:
            print(f"  epoch {epoch:3d} train={tr:.4f} val={va:.4f}"
                  f"{' *' if improved else ''}", flush=True)
        if bad >= CNN_PATIENCE:
            print(f"  early stopping at epoch {epoch} (best val={best_val:.4f})")
            break

    assert best_state is not None
    ckpt = MODELS_DIR / f"best_{run_tag}.pt"
    torch.save({"model_name": model_name, "feature_names": feat_names,
                "state_dict": best_state, "seed": seed, "augment": bool(augment)},
               ckpt)
    joblib.dump(feat_scaler, MODELS_DIR / f"feat_scaler_{run_tag}.joblib")
    joblib.dump(target_scaler, MODELS_DIR / f"target_scaler_{run_tag}.joblib")
    pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)
    print(f"Saved {ckpt} + scalers + {run_dir/'train_history.csv'} "
          f"({time.time()-t0:.0f}s, best val loss {best_val:.4f})")
    return ckpt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["hybrid", "image_only", "mlp"])
    ap.add_argument("--features", default="final", choices=["final", "all25"])
    ap.add_argument("--seed", type=int, default=CNN_SEED,
                    help="seed for init + batch order (default CNN_SEED=%d)" % CNN_SEED)
    ap.add_argument("--no-augment", action="store_true",
                    help="disable the train-only flip augmentation")
    args = ap.parse_args()
    CNN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_model(args.model, args.features, seed=args.seed,
                augment=False if args.no_augment else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
