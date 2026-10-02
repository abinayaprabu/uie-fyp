#!/usr/bin/env python3
"""Verify the FINAL feature-guided checkpoint before any test evaluation.

A training process exiting with code 0 is not evidence that training worked.
This script checks the things the specification requires (section 14-15) and
writes the result to ``results/enhancement/<run>/model_verification.json``:

  1. the checkpoint file exists in BOTH locations (models/ and the durable
     results/ copy) and the two are byte-identical (sha256);
  2. it loads with ``torch.load`` and the architecture in it matches the code;
  3. the feature list inside the checkpoint is EXACTLY the statistically
     selected list in results/feature_selection/selected_features.txt, and the
     scaler it stores is finite and non-degenerate;
  4. a real test image runs through the model: correct output shape, valid
     range, no NaN/Inf;
  5. the feature branch actually matters after training: the output changes
     when the selected features are zeroed or shuffled (if it does not, the
     guidance is broken and the run must not be reported);
  6. the training history exists, has the recorded best epoch, and that epoch's
     validation SSIM equals the value stored in the checkpoint.

Usage:
    python scripts/verify_final_model.py --run enh224_featguided
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn.dataset import letterbox  # noqa: E402
from cnn.dataset_pairs import split_ids, to_tensor  # noqa: E402
from cnn.feature_guided.model import FeatureGuidedEnhancer  # noqa: E402
from src.config import (  # noqa: E402
    ENHANCEMENT_RESULTS_DIR, FEATURE_RESULTS_DIR, HYBRID_RUN_GUIDED,
    MODELS_DIR, PREPROCESSED_DIR, RESULTS_DIR,
)

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=HYBRID_RUN_GUIDED)
    args = ap.parse_args()
    run = args.run

    print("=" * 78)
    print(f"FINAL MODEL VERIFICATION -- run {run}")
    print("=" * 78)

    ckpt_path = MODELS_DIR / f"best_{run}.pt"
    durable = ENHANCEMENT_RESULTS_DIR / run / f"best_{run}.pt"
    out_dir = ENHANCEMENT_RESULTS_DIR / run
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1] checkpoint files")
    check("models/ checkpoint exists", ckpt_path.exists(), str(ckpt_path))
    check("durable results/ copy exists", durable.exists(), str(durable))
    if not (ckpt_path.exists() and durable.exists()):
        print("\nRESULT: FAILED -- no checkpoint to verify")
        return 1
    h1, h2 = sha256(ckpt_path), sha256(durable)
    check("byte-identical copies", h1 == h2, f"sha256 {h1[:16]}...")

    print("\n[2] loads and matches the code")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = FeatureGuidedEnhancer(
        n_features=ckpt["arch"]["n_features"],
        use_features=ckpt["use_features"],
        encoder_widths=tuple(ckpt["arch"]["encoder_widths"]),
        decoder_widths=tuple(ckpt["arch"]["decoder_widths"]),
    )
    try:
        model.load_state_dict(ckpt["state_dict"], strict=True)
        load_ok, load_msg = True, "strict load"
    except Exception as exc:                      # pragma: no cover - failure path
        load_ok, load_msg = False, str(exc)[:140]
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    check("state_dict loads strictly", load_ok, load_msg)
    check("use_features is True (this is the proposed model)",
          bool(ckpt["use_features"]))
    check("architecture matches", tuple(ckpt["arch"]["encoder_widths"]) == (32, 64, 128, 256)
          and tuple(ckpt["arch"]["decoder_widths"]) == (128, 64, 32, 16),
          f"{ckpt['arch']['encoder_widths']} / {ckpt['arch']['decoder_widths']}")
    check("parameter count matches the recorded value",
          n_params == int(ckpt["arch"]["n_params"]),
          f"{n_params:,} vs {ckpt['arch']['n_params']:,} recorded")

    print("\n[3] feature list == the statistically selected list")
    sel_txt = RESULTS_DIR / "feature_selection" / "selected_features.txt"
    names_in_ckpt = list(ckpt["feature_names"])
    names_on_disk = [ln.split(". ", 1)[1].strip()
                     for ln in sel_txt.read_text().splitlines()
                     if ln.strip() and not ln.startswith("#")]
    check("checkpoint features == results/feature_selection/selected_features.txt",
          names_in_ckpt == names_on_disk, f"{names_in_ckpt}")
    check("number of features is the statistical result (not forced to 14)",
          len(names_in_ckpt) == 10, f"n_features={len(names_in_ckpt)}")
    mean = np.asarray(ckpt["scaler_mean"], dtype=float)
    scale = np.asarray(ckpt["scaler_scale"], dtype=float)
    check("scaler is finite and non-degenerate",
          np.isfinite(mean).all() and np.isfinite(scale).all()
          and (scale > 0).all() and scale.size == len(names_in_ckpt),
          f"scale range [{scale.min():.4f}, {scale.max():.4f}]")

    print("\n[4] forward pass on a real test image")
    name = split_ids("test")[0]
    pre = cv2.imread(str(PREPROCESSED_DIR / name), cv2.IMREAD_COLOR)
    x = to_tensor(letterbox(cv2.cvtColor(pre, cv2.COLOR_BGR2RGB))).unsqueeze(0)
    q = pd.read_csv(FEATURE_RESULTS_DIR / "feature_quality_dataset.csv").set_index("image_name")
    raw = q.loc[name, names_in_ckpt].to_numpy(dtype=float)
    feats = torch.from_numpy(((raw - mean) / scale).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        out = model(x, feats)
    check("output shape is (1, 3, 224, 224)", tuple(out.shape) == (1, 3, 224, 224),
          str(tuple(out.shape)))
    check("output in [0, 1]", float(out.min()) >= 0.0 and float(out.max()) <= 1.0,
          f"[{float(out.min()):.3f}, {float(out.max()):.3f}]")
    check("no NaN / Inf", bool(torch.isfinite(out).all()))

    print("\n[5] the feature branch actually matters after training")
    with torch.no_grad():
        out_zero = model(x, torch.zeros_like(feats))
        perm = torch.randperm(feats.shape[1])
        out_shuf = model(x, feats[:, perm])
    d_zero = float((out - out_zero).abs().max())
    d_shuf = float((out - out_shuf).abs().max())
    check("zeroing the features changes the output", d_zero > 1e-6, f"max |delta| {d_zero:.3e}")
    check("shuffling the features changes the output", d_shuf > 1e-6, f"max |delta| {d_shuf:.3e}")

    print("\n[6] training history agrees with the checkpoint")
    hist_path = out_dir / "train_history.csv"
    check("history exists", hist_path.exists(), str(hist_path))
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        best_epoch = int(ckpt["best_epoch"])
        check("best epoch recorded in history",
              best_epoch in hist["epoch"].to_numpy(),
              f"best epoch {best_epoch} of {int(hist['epoch'].max())} completed")
        row = hist.loc[hist["epoch"] == best_epoch]
        if len(row) and pd.notna(row.iloc[0]["val_ssim"]):
            check("history val SSIM == checkpoint val SSIM",
                  abs(float(row.iloc[0]["val_ssim"]) - float(ckpt["best_val_ssim"])) < 1e-6,
                  f"{float(row.iloc[0]['val_ssim']):.6f} vs {float(ckpt['best_val_ssim']):.6f}")

    summary = {
        "run": run,
        "checkpoint": str(ckpt_path),
        "durable_copy": str(durable),
        "sha256_prefix": h1[:16],
        "n_parameters": n_params,
        "n_features": len(names_in_ckpt),
        "feature_names": names_in_ckpt,
        "best_epoch": int(ckpt["best_epoch"]),
        "best_val_ssim": float(ckpt["best_val_ssim"]),
        "best_val_psnr": float(ckpt["best_val_psnr"]),
        "epochs_completed": int(ckpt.get("epochs_completed", -1)),
        "feature_effect_zeroed_max_delta": d_zero,
        "feature_effect_shuffled_max_delta": d_shuf,
        "train_ids_fingerprint": ckpt["train_ids_fingerprint"],
        "val_ids_fingerprint": ckpt["val_ids_fingerprint"],
        "test_ids_fingerprint": ckpt["test_ids_fingerprint"],
        "checks_failed": failures,
    }
    with open(out_dir / "model_verification.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_dir/'model_verification.json'}")

    print("\n" + "=" * 78)
    if failures:
        print(f"RESULT: {len(failures)} CHECK(S) FAILED -> {failures}")
        return 1
    print("RESULT: FINAL MODEL VERIFIED (checkpoint, features, forward pass, "
          "feature dependence, history)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
