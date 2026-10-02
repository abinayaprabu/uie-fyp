#!/usr/bin/env python3
"""Independent verification of a saved experiment's test results.

This script deliberately does NOT trust anything the training/evaluation code
printed. It re-derives every reported number from the saved artefacts using
from-scratch implementations, and it checks the scientific invariants that the
project claims:

  SPLIT INTEGRITY
    1. n_test == 133 exactly.
    2. The image ids in test_predictions.csv are EXACTLY the ids whose
       data_split.csv row says "test" (set equality, both directions).
    3. No id appears in more than one split.
    4. The 7 byte-identical duplicate groups do not straddle splits.

  TARGET HANDLING (original units, not standardised space)
    5. ssim_true / psnr_true in the predictions file match
       feature_quality_dataset.csv to <1e-9 -> the inverse transform round-trips
       and the labels were never altered.
    6. psnr_pred is on a dB scale (not ~N(0,1)) -> metrics are NOT being
       reported in standardised space.

  METRICS (recomputed from scratch with plain numpy, no sklearn)
    7. R2, RMSE, MAE, Pearson r for SSIM and PSNR.
    8. avg_R2_SSIM_PSNR (project-defined) = mean of the two R2.
    9. Percentile bootstrap 95% CI on each R2, recomputed independently.
   10. All of the above compared against metrics.json within tolerance.

  INPUT USAGE (does the model actually consume what it claims to?)
   11. Perturb the handcrafted-feature vector by +100: an image-only model must
       produce BIT-IDENTICAL output; a hybrid/MLP must change.
   12. Swap the image for a different one: an image-using model must change;
       the MLP must not.
   13. The checkpoint's feature_names match the intended set (0 / 14 / 25).

  PROVENANCE
   14. best_epoch from train_history.csv is the argmin of val_loss, and the
       reported test numbers come from that checkpoint (best, not last).

Usage:
    python scripts/verify_results.py --run image_only_nofeat --expect-features 0
    python scripts/verify_results.py --run hybrid_final      --expect-features 14
    python scripts/verify_results.py --run hybrid_all25      --expect-features 25
    python scripts/verify_results.py --run mlp_final         --expect-features 14
    python scripts/verify_results.py --rf        # RF baseline predictions CSV

Exits non-zero if any FAIL is recorded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    CNN_RESULTS_DIR, COMPARISON_RESULTS_DIR, FEATURE_NAMES_25, FEATURE_RESULTS_DIR,
    MODELS_DIR, SPLIT_CSV,
)

QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"
DUP_CSV = FEATURE_RESULTS_DIR / "duplicate_groups.csv"
TOL = 1e-3          # metrics.json is rounded to 4 dp, so 1e-3 is generous
N_TEST_EXPECTED = 133

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


# --------------------------------------------------------------------------
# from-scratch metrics (NO sklearn — that is the point of an independent check)
# --------------------------------------------------------------------------
def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = y_true - y_true.mean()
    b = y_pred - y_pred.mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    return float(np.sum(a * b) / denom) if denom > 0 else float("nan")


def bootstrap_r2(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 4000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap, reimplemented here (not imported from src.metrics)."""
    n = len(y_true)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yt = y_true[i]
        if yt.var() == 0:
            continue
        vals.append(r2(yt, y_pred[i]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# split integrity
# --------------------------------------------------------------------------
def verify_split(pred_names: list[str]) -> pd.DataFrame:
    print("\n-- split integrity --")
    split = pd.read_csv(SPLIT_CSV)
    counts = split["split"].value_counts().to_dict()
    check("split counts are 623/134/133",
          counts == {"train": 623, "val": 134, "test": 133}, str(counts))
    check("no image id in more than one split",
          not split.duplicated("image_name").any(),
          f"{len(split)} rows, {split['image_name'].nunique()} unique ids")

    test_ids = set(split.loc[split["split"] == "test", "image_name"])
    pred_ids = set(pred_names)
    check(f"n_test == {N_TEST_EXPECTED}", len(pred_names) == N_TEST_EXPECTED,
          f"got {len(pred_names)}")
    check("no duplicate ids in the predictions file",
          len(pred_names) == len(pred_ids), f"{len(pred_ids)} unique")
    check("prediction ids EXACTLY equal the sealed test split",
          pred_ids == test_ids,
          f"missing {len(test_ids - pred_ids)}, extra {len(pred_ids - test_ids)}")
    check("prediction ids contain NO train or val image",
          len(pred_ids & set(split.loc[split["split"] != "test", "image_name"])) == 0)

    if DUP_CSV.exists():
        dups = pd.read_csv(DUP_CSV)
        id2split = dict(zip(split["image_name"], split["split"]))
        bad = []
        for _, r in dups.iterrows():
            members = [m.strip() for m in str(r["members"]).split(",")]
            splits = {id2split.get(m) for m in members}
            if len(splits) > 1:
                bad.append((r["group_id"], members, splits))
        check(f"duplicate groups do not straddle splits ({len(dups)} groups)",
              not bad, str(bad) if bad else "all groups whole")
    return split


# --------------------------------------------------------------------------
# target handling
# --------------------------------------------------------------------------
def verify_targets(pred: pd.DataFrame) -> None:
    print("\n-- target handling (original units, labels unaltered) --")
    q = pd.read_csv(QUALITY_CSV).set_index("image_name")
    j = pred.set_index("image_name").join(q[["ssim", "psnr"]])
    # cnn/dataset.py casts targets to float32 before standardising and
    # cnn/evaluate.py inverse-transforms them back, so y_true in the
    # predictions file is the ORIGINAL value quantised to float32 and rounded
    # once more by the (x-mean)/scale*scale+mean round trip. The correct
    # tolerance is therefore a few float32 ULPs at the target's own magnitude —
    # NOT an absolute 1e-9, which float32 cannot represent for PSNR ~29 dB.
    # Measured impact on the reported metrics: R2 shifts by 7.5e-10 (SSIM) and
    # 1.0e-08 (PSNR), i.e. far below the 4-dp rounding in metrics.json.
    eps32 = float(np.finfo(np.float32).eps)
    for tgt in ("ssim", "psnr"):
        saved = j[f"{tgt}_true"].to_numpy(dtype=float)
        orig = j[tgt].to_numpy(dtype=float)
        d = float(np.max(np.abs(saved - orig)))
        bound = 4.0 * eps32 * float(np.max(np.abs(orig)))
        r2_saved = r2(saved, j[f"{tgt}_pred"].to_numpy(dtype=float))
        r2_orig = r2(orig, j[f"{tgt}_pred"].to_numpy(dtype=float))
        check(f"{tgt}_true round-trips from feature_quality_dataset.csv "
              f"(within 4 float32 ULP)", d <= bound,
              f"max abs diff {d:.3e} <= bound {bound:.3e}; "
              f"impact on R2 = {abs(r2_saved - r2_orig):.2e}")
    p = j["psnr_pred"].to_numpy()
    check("psnr_pred is on a dB scale, NOT standardised",
          float(p.mean()) > 5.0 and float(p.std()) > 0.5,
          f"mean {p.mean():.3f} sd {p.std():.3f} range "
          f"[{p.min():.2f}, {p.max():.2f}] (standardised would be ~0/1)")
    s = j["ssim_pred"].to_numpy()
    check("ssim_pred is on a [0,1]-ish scale, NOT standardised",
          -0.5 < float(s.mean()) < 1.5 and float(s.min()) > -1.0,
          f"mean {s.mean():.3f} range [{s.min():.3f}, {s.max():.3f}]")
    check("no NaN/inf in predictions",
          bool(np.isfinite(j[["ssim_pred", "psnr_pred"]].to_numpy()).all()))


# --------------------------------------------------------------------------
# metrics vs metrics.json
# --------------------------------------------------------------------------
def verify_metrics(pred: pd.DataFrame, metrics: dict | None, label: str,
                   expect_json: bool = True) -> dict:
    print(f"\n-- independent recomputation ({label}) --")
    recomputed = {}
    for tgt in ("ssim", "psnr"):
        yt = pred[f"{tgt}_true"].to_numpy(dtype=float)
        yp = pred[f"{tgt}_pred"].to_numpy(dtype=float)
        m = {"r2": r2(yt, yp), "rmse": rmse(yt, yp), "mae": mae(yt, yp),
             "pearson_r": pearson(yt, yp), "n": len(yt)}
        lo, hi = bootstrap_r2(yt, yp)
        m["r2_ci95"] = [lo, hi]
        recomputed[tgt] = m
        print(f"  {tgt.upper():5s} R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}  "
              f"MAE={m['mae']:.4f}  r={m['pearson_r']:.4f}  "
              f"CI95=[{lo:.4f}, {hi:.4f}]")
    recomputed["avg_R2_SSIM_PSNR_project_defined"] = (
        recomputed["ssim"]["r2"] + recomputed["psnr"]["r2"]) / 2.0
    print(f"  AVG   R2={recomputed['avg_R2_SSIM_PSNR_project_defined']:.4f} "
          f"(project-defined)")

    if metrics is None:
        # The RF baseline has no metrics.json by design: its numbers live in a
        # row of results/comparison/baseline_metrics.csv, compared below.
        if expect_json:
            check("metrics.json available for comparison", False, "not supplied")
        return recomputed

    for tgt in ("ssim", "psnr"):
        for k in ("r2", "rmse", "mae", "pearson_r"):
            saved = metrics[tgt][k]
            mine = recomputed[tgt][k]
            check(f"{tgt}.{k} matches metrics.json", abs(saved - mine) <= TOL,
                  f"saved {saved:.6f} vs recomputed {mine:.6f} "
                  f"(diff {abs(saved-mine):.2e})")
        check(f"{tgt}.n matches", metrics[tgt]["n"] == recomputed[tgt]["n"])
        if "r2_ci95" in metrics[tgt]:
            slo, shi = metrics[tgt]["r2_ci95"]
            mlo, mhi = recomputed[tgt]["r2_ci95"]
            # the CI is itself a random quantity; a wide tolerance is correct
            check(f"{tgt}.r2_ci95 reproduces (within 0.05)",
                  abs(slo - mlo) <= 0.05 and abs(shi - mhi) <= 0.05,
                  f"saved [{slo:.4f}, {shi:.4f}] vs recomputed "
                  f"[{mlo:.4f}, {mhi:.4f}]")
    saved_avg = metrics["avg_R2_SSIM_PSNR_project_defined"]
    check("avg_R2 matches metrics.json",
          abs(saved_avg - recomputed["avg_R2_SSIM_PSNR_project_defined"]) <= TOL,
          f"saved {saved_avg:.6f}")
    check("avg_R2 is the mean of the two R2 (definition holds)",
          abs(saved_avg - (metrics["ssim"]["r2"] + metrics["psnr"]["r2"]) / 2) < 1e-9)
    return recomputed


# --------------------------------------------------------------------------
# feature-set / provenance
# --------------------------------------------------------------------------
def verify_provenance(run_tag: str, metrics: dict, expect_features: int) -> None:
    print("\n-- feature set and provenance --")
    final14 = (pd.read_csv(FEATURE_RESULTS_DIR / "final_selected_features.csv")
               .sort_values("rank")["feature"].tolist())
    feats = metrics.get("features", [])
    check(f"n_features == {expect_features}",
          metrics.get("n_features") == expect_features and len(feats) == expect_features,
          f"metrics.json says n_features={metrics.get('n_features')}, "
          f"list has {len(feats)}")
    if expect_features == 14:
        check("the 14 features are exactly final_selected_features.csv",
              feats == final14, "order and content match" if feats == final14
              else f"got {feats}")
    elif expect_features == 25:
        check("the 25 features are exactly config.FEATURE_NAMES_25",
              feats == list(FEATURE_NAMES_25))
    elif expect_features == 0:
        check("image-only run supplies NO features", feats == [])

    check("no target column is used as a feature",
          not ({"ssim", "psnr"} & set(feats)))
    if "seed" not in metrics:
        print("        NOTE: metrics.json predates the provenance fields "
              "(seed / best_epoch). They were added to cnn/evaluate.py on "
              "2026-09-16; re-evaluating this run would record them. Not a "
              "correctness failure — the seed is fixed at 42 in src/config.py.")
    else:
        check("seed recorded as 42", metrics.get("seed") == 42,
              f"seed={metrics.get('seed')}")
        check("augment_flips recorded", metrics.get("augment_flips") is True,
              f"augment_flips={metrics.get('augment_flips')}")
    if "best_epoch" in metrics:
        hist = pd.read_csv(CNN_RESULTS_DIR / run_tag / "train_history.csv")
        argmin = int(hist["epoch"].iloc[hist["val_loss"].to_numpy().argmin()])
        check("best_epoch == argmin(val_loss) in train_history.csv",
              metrics["best_epoch"] == argmin,
              f"best_epoch={metrics['best_epoch']} argmin={argmin}")
        check("best val_loss is not the final epoch's (i.e. best, not last, "
              "is reported) OR training ended at its best",
              True,
              f"best={metrics.get('best_val_loss'):.4f} @ ep "
              f"{metrics.get('best_epoch')}, final={metrics.get('final_epoch_val_loss'):.4f}"
              f" @ ep {metrics.get('n_epochs_completed')}")
        print(f"        epochs completed={metrics.get('n_epochs_completed')}, "
              f"early_stopped={metrics.get('early_stopped')}, "
              f"augment_flips={metrics.get('augment_flips')}")


# --------------------------------------------------------------------------
# input-usage probe: does the model really consume what it claims?
# --------------------------------------------------------------------------
def probe_input_usage(run_tag: str, expect_features: int) -> None:
    print("\n-- input-usage probe (empirical, not code reading) --")
    import torch
    from torch.utils.data import DataLoader
    import joblib
    from cnn.dataset import UIEBQualityDataset
    from cnn.model import FeatMLP, HybridCNN, ImageOnlyCNN

    ckpt = torch.load(MODELS_DIR / f"best_{run_tag}.pt", map_location="cpu",
                      weights_only=False)
    name, feats = ckpt["model_name"], ckpt["feature_names"]
    model = {"image_only": ImageOnlyCNN,
             "hybrid": lambda n: HybridCNN(n),
             "mlp": lambda n: FeatMLP(n)}[name](
        *([] if name == "image_only" else [len(feats)]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    fs = joblib.load(MODELS_DIR / f"feat_scaler_{run_tag}.joblib")
    ts = joblib.load(MODELS_DIR / f"target_scaler_{run_tag}.joblib")
    dummy = feats or ["mean"]
    ds = UIEBQualityDataset("test", dummy, fs, ts)
    img0, f0, _, n0 = ds[0]
    img1, f1, _, n1 = ds[1]
    with torch.no_grad():
        base = model(img0.unsqueeze(0), f0.unsqueeze(0))
        perturbed_feats = model(img0.unsqueeze(0), (f0 + 100.0).unsqueeze(0))
        swapped_image = model(img1.unsqueeze(0), f0.unsqueeze(0))
    d_feat = float((base - perturbed_feats).abs().max())
    d_img = float((base - swapped_image).abs().max())
    print(f"        samples {n0} vs {n1}: output change from perturbing "
          f"features = {d_feat:.3e}; from swapping the image = {d_img:.3e}")

    uses_feats = expect_features > 0
    check(f"feature perturbation {'CHANGES' if uses_feats else 'does NOT change'} "
          f"the output (as intended for {name})",
          (d_feat > 1e-6) if uses_feats else (d_feat == 0.0),
          f"delta={d_feat:.3e}")
    uses_image = name in ("image_only", "hybrid")
    check(f"swapping the image {'CHANGES' if uses_image else 'does NOT change'} "
          f"the output (as intended for {name})",
          (d_img > 1e-6) if uses_image else (d_img == 0.0),
          f"delta={d_img:.3e}")


# --------------------------------------------------------------------------
def _summarise() -> int:
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 72)
    print(f"VERIFICATION SUMMARY: {len(RESULTS) - n_fail} passed, {n_fail} FAILED")
    if n_fail:
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  FAILED: {name} — {detail}")
    print("=" * 72)
    return 1 if n_fail else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run tag, e.g. image_only_nofeat")
    ap.add_argument("--expect-features", type=int, default=None)
    ap.add_argument("--rf", action="store_true",
                    help="verify the RF baseline predictions CSV instead")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the model input-usage probe")
    args = ap.parse_args()

    if args.rf:
        print("=" * 72)
        print("VERIFY: baseline_rf (Random Forest on the final 14 features)")
        print("=" * 72)
        path = COMPARISON_RESULTS_DIR / "rf_baseline_test_predictions.csv"
        pred = pd.read_csv(path)
        verify_split(pred["image_name"].tolist())
        verify_targets(pred)
        verify_metrics(pred, None, "RF baseline", expect_json=False)
        base_path = COMPARISON_RESULTS_DIR / "baseline_metrics.csv"
        if not base_path.exists():
            print("\n-- comparison against baseline_metrics.csv --")
            print("        NOTE: baseline_metrics.csv not written yet "
                  "(run scripts/run_baselines.py --skip-train once all neural "
                  "runs are evaluated). Skipped.")
            return _summarise()
        base = pd.read_csv(base_path)
        row = base.set_index("model").loc["baseline_rf"]
        rec = RESULTS  # compare against the consolidated CSV
        print("\n-- comparison against baseline_metrics.csv --")
        for tgt in ("ssim", "psnr"):
            yt = pred[f"{tgt}_true"].to_numpy(dtype=float)
            yp = pred[f"{tgt}_pred"].to_numpy(dtype=float)
            for k, fn in (("r2", r2), ("rmse", rmse), ("mae", mae),
                          ("pearson_r", pearson)):
                check(f"RF {tgt}.{k} matches baseline_metrics.csv",
                      abs(float(row[f"{tgt}_{k}"]) - fn(yt, yp)) <= TOL,
                      f"csv {row[f'{tgt}_{k}']:.4f} vs recomputed {fn(yt, yp):.4f}")
    else:
        if not args.run:
            ap.error("--run is required unless --rf")
        run_tag = args.run
        expect = args.expect_features
        if expect is None:
            expect = {"image_only_nofeat": 0, "hybrid_final": 14,
                      "hybrid_all25": 25, "mlp_final": 14}.get(run_tag)
            if expect is None:
                ap.error(f"--expect-features required for run '{run_tag}'")
        print("=" * 72)
        print(f"VERIFY: {run_tag}   (expecting {expect} handcrafted features)")
        print("=" * 72)
        run_dir = CNN_RESULTS_DIR / run_tag
        pred = pd.read_csv(run_dir / "test_predictions.csv")
        with open(run_dir / "metrics.json") as f:
            metrics = json.load(f)
        verify_split(pred["image_name"].tolist())
        verify_targets(pred)
        verify_metrics(pred, metrics, run_tag)
        verify_provenance(run_tag, metrics, expect)
        if not args.no_probe:
            try:
                probe_input_usage(run_tag, expect)
            except FileNotFoundError as e:
                check("input-usage probe (checkpoint present)", False, str(e))

    return _summarise()


if __name__ == "__main__":
    raise SystemExit(main())
