#!/usr/bin/env python3
"""Independent verification of the enhancement experiment.

Nothing here trusts a number printed during training. Every reported figure is
re-derived from artefacts ON DISK — the enhanced PNGs, the preprocessed PNGs,
the reference PNGs, the checkpoint, the history CSV — and compared against what
``scripts/evaluate_enhancement.py`` saved.

Checks, grouped by the question they answer:

  DO THE ENHANCED IMAGES EXIST AND COVER THE RIGHT IMAGES?
    1. dataset/enhanced-test/ holds exactly 133 files, one per sealed test id,
       with no extras and none missing.
    2. Those ids are EXACTLY the ids whose data_split.csv row says "test".
    3. Each enhanced PNG is uint8, 3-channel, and has the same shape as both the
       preprocessed input and the aligned reference.

  WAS THE TEST SET KEPT OUT OF TRAINING?
    4. The checkpoint's recorded train-id fingerprint equals the fingerprint of
       the frozen train split recomputed from data_split.csv.
    5. Same for the val fingerprint.
    6. Train, val and test id sets are pairwise disjoint.
    7. The enhanced PNGs were all written AFTER the checkpoint was saved (so the
       reported images came from that checkpoint, not an older one).

  WAS THE FROZEN CONFIG ACTUALLY USED, AND THE RIGHT CHECKPOINT SELECTED?
    8. Every value in the checkpoint's config block equals the frozen Config A
       in src/config.py (crop, batch, lr, loss, epochs, val_every, patience,
       augmentation flags, base channels, parameter count).
    9. best_epoch == argmax(val_ssim) over the validation rows of
       train_history.csv, and epochs_completed == len(history).

  ARE THE METRICS RIGHT?
   10. SSIM/PSNR recomputed from the saved PNGs with src.iqa must match
       test_per_image.csv to 1e-12 / 1e-9.
   11. PSNR recomputed a second time from the raw formula (no scikit-image) must
       match to 1e-9.
   12. SSIM recomputed a second time from a from-scratch uniform-window
       implementation must match to 1e-9.
   13. The classical baseline recomputed here must match the committed
       feature_quality_dataset.csv labels exactly — proving this script uses the
       same metric definition that produced the project's targets.
   14. Aggregates in metrics.json must match a fresh recomputation.

  DID THE NETWORK ACTUALLY DO SOMETHING, WITHOUT CHEATING?
   15. Enhanced images differ from the preprocessed input (the network is not an
       identity mapping).
   16. No enhanced image is byte-identical to its reference, and the maximum
       per-image SSIM against the reference is reported — a value pinned at 1.0
       would indicate the target leaked into the input path.

Usage:
    python scripts/verify_enhancement.py [--run unet_128]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn.dataset_pairs import aligned_reference, ids_fingerprint, split_ids  # noqa: E402
from src.config import (  # noqa: E402
    ENH_AUGMENT_GEOMETRIC, ENH_BASE_CHANNELS, ENH_BATCH_SIZE, ENH_CROP_SIZE,
    ENH_EXPECTED_PARAMS, ENH_LOSS, ENH_LR, ENH_MAX_EPOCHS, ENH_PATIENCE,
    ENH_RUN_TAG, ENH_SEED, ENH_VAL_EVERY, ENHANCED_DIR, ENHANCEMENT_RESULTS_DIR,
    FEATURE_RESULTS_DIR, MODELS_DIR, PREPROCESSED_DIR,
)
from src.iqa import compute_ssim_psnr, psnr_from_scratch, ssim_from_scratch  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=ENH_RUN_TAG)
    ap.add_argument("--limit-scratch", type=int, default=20,
                    help="how many images get the slow from-scratch SSIM recheck")
    args = ap.parse_args()

    out_dir = ENHANCEMENT_RESULTS_DIR / args.run
    print("=" * 74)
    print(f"INDEPENDENT VERIFICATION: enhancement run '{args.run}'")
    print("=" * 74)

    per_image_path = out_dir / "test_per_image.csv"
    metrics_path = out_dir / "metrics.json"
    ckpt_path = MODELS_DIR / f"best_{args.run}.pt"
    hist_path = out_dir / "train_history.csv"
    for p in (per_image_path, metrics_path, ckpt_path, hist_path):
        if not p.exists():
            print(f"FATAL: required artefact missing: {p}")
            return 2

    per = pd.read_csv(per_image_path)
    with open(metrics_path) as f:
        met = json.load(f)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hist = pd.read_csv(hist_path)

    test_ids = split_ids("test")
    train_ids = split_ids("train")
    val_ids = split_ids("val")

    # ---- 1-3: enhanced images on disk -------------------------------------
    print("\n-- enhanced images exist and cover the sealed test split --")
    on_disk = sorted(p.name for p in ENHANCED_DIR.glob("*.png"))
    check(f"exactly {len(test_ids)} enhanced PNGs on disk",
          len(on_disk) == len(test_ids), f"found {len(on_disk)}")
    check("filenames are EXACTLY the sealed test ids (no extras, none missing)",
          set(on_disk) == set(test_ids),
          f"extra {len(set(on_disk)-set(test_ids))}, "
          f"missing {len(set(test_ids)-set(on_disk))}")
    check("per-image CSV covers exactly the test ids",
          set(per["image_name"]) == set(test_ids), f"{len(per)} rows")

    shapes_ok, dtypes_ok, ch_ok = [], [], []
    max_ssim_ref = 0.0
    ident_to_ref, ident_to_pre = [], []
    d_pre, d_ref = [], []
    for name in test_ids:
        enh = cv2.imread(str(ENHANCED_DIR / name), cv2.IMREAD_COLOR)
        pre = cv2.imread(str(PREPROCESSED_DIR / name), cv2.IMREAD_COLOR)
        ref = aligned_reference(name)
        shapes_ok.append(enh.shape == pre.shape == ref.shape)
        dtypes_ok.append(enh.dtype == np.uint8)
        ch_ok.append(enh.ndim == 3 and enh.shape[2] == 3)
        if np.array_equal(enh, ref):
            ident_to_ref.append(name)
        if np.array_equal(enh, pre):
            ident_to_pre.append(name)
        d_pre.append(float(np.mean(np.abs(enh.astype(np.int16) - pre.astype(np.int16)))))
        d_ref.append(float(np.mean(np.abs(enh.astype(np.int16) - ref.astype(np.int16)))))
        s_row = float(per.loc[per["image_name"] == name, "ssim_unet"].iloc[0])
        max_ssim_ref = max(max_ssim_ref, s_row)
    check("every enhanced image has the same shape as its input AND reference",
          all(shapes_ok), f"{sum(shapes_ok)}/{len(shapes_ok)}")
    check("every enhanced image is uint8", all(dtypes_ok))
    check("every enhanced image is 3-channel", all(ch_ok))

    # ---- 4-7: leakage -----------------------------------------------------
    print("\n-- the test set was kept out of training --")
    check("checkpoint train-id fingerprint matches the frozen train split",
          ckpt.get("train_ids_fingerprint") == ids_fingerprint(train_ids),
          f"ckpt {ckpt.get('train_ids_fingerprint')} vs "
          f"recomputed {ids_fingerprint(train_ids)}")
    check("checkpoint val-id fingerprint matches the frozen val split",
          ckpt.get("val_ids_fingerprint") == ids_fingerprint(val_ids))
    check("checkpoint n_train / n_val are 623 / 134",
          ckpt.get("n_train") == 623 and ckpt.get("n_val") == 134,
          f"{ckpt.get('n_train')} / {ckpt.get('n_val')}")
    check("train, val and test id sets are pairwise disjoint",
          not (set(train_ids) & set(val_ids))
          and not (set(train_ids) & set(test_ids))
          and not (set(val_ids) & set(test_ids)))
    ckpt_mtime = ckpt_path.stat().st_mtime
    older = [n for n in test_ids
             if (ENHANCED_DIR / n).stat().st_mtime < ckpt_mtime]
    check("all enhanced PNGs were written AFTER the best checkpoint was saved",
          not older, f"{len(older)} stale" if older else
          f"checkpoint saved, then all {len(test_ids)} images written")

    # ---- 8-9: frozen config + checkpoint selection ------------------------
    print("\n-- frozen Config A was used, and the best checkpoint was selected --")
    cfg = ckpt.get("config", {})
    frozen = {"crop": ENH_CROP_SIZE, "batch_size": ENH_BATCH_SIZE, "lr": ENH_LR,
              "weight_decay": 0.0, "loss": ENH_LOSS, "max_epochs": ENH_MAX_EPOCHS,
              "val_every": ENH_VAL_EVERY, "patience_validations": ENH_PATIENCE,
              "augment_geometric": bool(ENH_AUGMENT_GEOMETRIC),
              "augment_photometric": False}
    drift = {k: (cfg.get(k), v) for k, v in frozen.items() if cfg.get(k) != v}
    check("checkpoint config equals the frozen Config A in src/config.py",
          not drift, str(drift) if drift else
          f"crop {cfg.get('crop')}, batch {cfg.get('batch_size')}, lr {cfg.get('lr')}, "
          f"loss {cfg.get('loss')}, max_epochs {cfg.get('max_epochs')}")
    check("parameter count equals the benchmarked 472,259",
          ckpt.get("arch", {}).get("n_params") == ENH_EXPECTED_PARAMS,
          f"{ckpt.get('arch', {}).get('n_params')}")
    check("seed recorded as 42", ckpt.get("seed") == ENH_SEED,
          f"seed={ckpt.get('seed')}")
    vrows = hist.dropna(subset=["val_ssim"])
    argmax = int(vrows.loc[vrows["val_ssim"].idxmax(), "epoch"])
    check("best_epoch == argmax(val_ssim) in train_history.csv",
          ckpt.get("best_epoch") == argmax,
          f"ckpt best_epoch={ckpt.get('best_epoch')}, history argmax={argmax}")
    # train_enhance.py stores FULL float64 precision in the checkpoint but
    # writes the history CSV rounded (val_ssim/val_l1 to 6 dp, val_psnr to 4 dp).
    # So the correct comparison is exact equality after applying the same
    # rounding — NOT a 1e-9 absolute tolerance, which is tighter than the CSV's
    # own rounding (max error 5e-7) and produced a spurious FAIL.
    brow = vrows.loc[vrows["val_ssim"].idxmax()]
    check("checkpoint best_val_ssim == that row's val_ssim (exact at 6-dp rounding)",
          round(float(ckpt["best_val_ssim"]), 6) == float(brow["val_ssim"]),
          f"round({ckpt['best_val_ssim']!r}, 6) = "
          f"{round(float(ckpt['best_val_ssim']), 6)} vs csv {float(brow['val_ssim'])}")
    check("checkpoint best_val_psnr == that row's val_psnr (exact at 4-dp rounding)",
          round(float(ckpt["best_val_psnr"]), 4) == float(brow["val_psnr"]),
          f"round({ckpt['best_val_psnr']!r}, 4) = "
          f"{round(float(ckpt['best_val_psnr']), 4)} vs csv {float(brow['val_psnr'])}")
    check("epochs_completed == number of history rows",
          ckpt.get("epochs_completed") == len(hist),
          f"{ckpt.get('epochs_completed')} vs {len(hist)}")
    check("epochs_completed <= the frozen max budget",
          int(ckpt.get("epochs_completed", 0)) <= ENH_MAX_EPOCHS,
          f"{ckpt.get('epochs_completed')} <= {ENH_MAX_EPOCHS}")

    # ---- 10-14: metrics ---------------------------------------------------
    print("\n-- metrics recomputed independently from the PNGs on disk --")
    worst_s = worst_p = 0.0
    worst_ss = worst_ps = 0.0
    n_scratch = 0
    for _, r in per.iterrows():
        name = r["image_name"]
        enh = cv2.imread(str(ENHANCED_DIR / name), cv2.IMREAD_COLOR)
        ref = aligned_reference(name)
        pre = cv2.imread(str(PREPROCESSED_DIR / name), cv2.IMREAD_COLOR)
        s, p = compute_ssim_psnr(enh, ref)
        worst_s = max(worst_s, abs(s - float(r["ssim_unet"])))
        worst_p = max(worst_p, abs(p - float(r["psnr_unet"])))
        sp, pp = compute_ssim_psnr(pre, ref)
        worst_ss = max(worst_ss, abs(sp - float(r["ssim_classical"])))
        worst_ps = max(worst_ps, abs(pp - float(r["psnr_classical"])))
        if n_scratch < args.limit_scratch:
            e_rgb = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)
            r_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
            d1 = abs(ssim_from_scratch(e_rgb, r_rgb) - s)
            d2 = abs(psnr_from_scratch(enh, ref) - p)
            check(f"from-scratch SSIM/PSNR agree for {name}",
                  d1 < 1e-9 and d2 < 1e-9, f"dSSIM {d1:.2e}, dPSNR {d2:.2e}")
            n_scratch += 1
    check("U-Net SSIM reproduced from disk for all 133 images",
          worst_s < 1e-12, f"max |diff| {worst_s:.2e}")
    check("U-Net PSNR reproduced from disk for all 133 images",
          worst_p < 1e-9, f"max |diff| {worst_p:.2e}")
    check("classical SSIM reproduced from disk for all 133 images",
          worst_ss < 1e-12, f"max |diff| {worst_ss:.2e}")
    check("classical PSNR reproduced from disk for all 133 images",
          worst_ps < 1e-9, f"max |diff| {worst_ps:.2e}")

    committed = pd.read_csv(FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"
                            ).set_index("image_name")
    dev_s = dev_p = 0.0
    for _, r in per.iterrows():
        n = r["image_name"]
        dev_s = max(dev_s, abs(float(r["ssim_classical"]) - float(committed.loc[n, "ssim"])))
        dev_p = max(dev_p, abs(float(r["psnr_classical"]) - float(committed.loc[n, "psnr"])))
    check("classical baseline equals the COMMITTED feature_quality_dataset.csv "
          "labels (same metric definition)",
          dev_s < 1e-12 and dev_p < 1e-9,
          f"max dSSIM {dev_s:.2e}, max dPSNR {dev_p:.2e}")

    print("\n-- aggregates in metrics.json --")
    for sysname, col_s, col_p in (("raw", "ssim_raw", "psnr_raw"),
                                  ("classical", "ssim_classical", "psnr_classical"),
                                  ("unet", "ssim_unet", "psnr_unet")):
        ms = float(per[col_s].mean()); mp = float(per[col_p].mean())
        js = met[sysname]["ssim"]["mean"]; jp = met[sysname]["psnr"]["mean"]
        check(f"metrics.json {sysname} mean SSIM/PSNR match a fresh recomputation",
              abs(ms - js) < 1e-6 and abs(mp - jp) < 1e-6,
              f"json {js:.6f}/{jp:.4f} vs recomputed {ms:.6f}/{mp:.4f}")
    d = per["ssim_unet"].to_numpy() - per["ssim_classical"].to_numpy()
    check("metrics.json SSIM delta matches",
          abs(float(d.mean()) - met["unet_minus_classical"]["ssim_mean_delta"]) < 1e-6,
          f"{met['unet_minus_classical']['ssim_mean_delta']:+.6f} vs {d.mean():+.6f}")
    check("metrics.json win/loss/tie counts match",
          met["unet_minus_classical"]["ssim_win"] == int((d > 1e-9).sum())
          and met["unet_minus_classical"]["ssim_loss"] == int((d < -1e-9).sum()),
          f"win {met['unet_minus_classical']['ssim_win']}, "
          f"loss {met['unet_minus_classical']['ssim_loss']}")
    check("metrics.json n_test == 133", met["n_test"] == 133, f"{met['n_test']}")

    # ---- 15-16: the network did something, without cheating ---------------
    print("\n-- the network actually transformed the image, without leaking --")
    check("enhanced images DIFFER from the preprocessed input (not identity)",
          not ident_to_pre and float(np.mean(d_pre)) > 0.5,
          f"mean |enh - preprocessed| = {np.mean(d_pre):.2f} grey levels; "
          f"identical files: {len(ident_to_pre)}")
    check("NO enhanced image is byte-identical to its reference (no target leak)",
          not ident_to_ref, f"identical: {len(ident_to_ref)}")
    check("max per-image SSIM vs reference is not pinned at 1.0",
          max_ssim_ref < 0.9999, f"max {max_ssim_ref:.6f}")
    print(f"        mean |enh - reference| = {np.mean(d_ref):.2f} grey levels")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 74)
    print(f"ENHANCEMENT VERIFICATION: {len(RESULTS) - n_fail} passed, {n_fail} FAILED")
    if n_fail:
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  FAILED: {name} — {detail}")
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
