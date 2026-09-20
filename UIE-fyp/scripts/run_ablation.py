#!/usr/bin/env python3
"""Ablation study — does feature ranking actually help the CNN?

  A: CNN image only                  (reuses image_only_nofeat test metrics)
  B: handcrafted features only, RF   (reuses baseline_rf test metrics;
                                      uses the FINAL SELECTED set, whose size
                                      is whatever subset evaluation chose -
                                      an earlier docstring said "8 features")
  C: CNN + ALL 25 features           (trains hybrid_all25 once, evaluates test)
  D: CNN + selected final features   (reuses hybrid_final test metrics)

Reuse (no retraining) for A/B/D is deliberate and documented: these runs are
identical to the baseline script's models, evaluated on the same test split.
Only C is trained here. Results -> results/comparison/ablation_results.csv.

    python scripts/run_ablation.py [--skip-train]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn.evaluate import evaluate_run  # noqa: E402
from cnn.train import train_model  # noqa: E402
from src.config import CNN_RESULTS_DIR, COMPARISON_RESULTS_DIR  # noqa: E402


def load_metrics(run_tag: str) -> dict:
    with open(CNN_RESULTS_DIR / run_tag / "metrics.json") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    base = pd.read_csv(COMPARISON_RESULTS_DIR / "baseline_metrics.csv")
    base = base.set_index("model").to_dict("index")

    if not args.skip_train:
        train_model("hybrid", "all25")
    met_c = evaluate_run("hybrid_all25")

    def neural_row(tag: str, metrics: dict, label: str) -> dict:
        return {"ablation": label, "model": tag,
                "n_features": metrics["n_features"], "n_test": metrics["n_test"],
                **{f"{t}_{k}": round(metrics[t][k], 4)
                   for t in ("ssim", "psnr") for k in ("r2", "rmse", "mae")},
                "avg_R2_SSIM_PSNR_project_defined": round(
                    metrics["avg_R2_SSIM_PSNR_project_defined"], 4)}

    rows = [
        {"ablation": "A_image_only", **{k: base["image_only_nofeat"][k]
         for k in ("n_test", "ssim_r2", "ssim_rmse", "ssim_mae", "psnr_r2",
                   "psnr_rmse", "psnr_mae", "avg_R2_SSIM_PSNR_project_defined")},
         "model": "image_only_nofeat", "n_features": 0},
        {"ablation": "B_features_only_RF", **{k: base["baseline_rf"][k]
         for k in ("n_test", "ssim_r2", "ssim_rmse", "ssim_mae", "psnr_r2",
                   "psnr_rmse", "psnr_mae", "avg_R2_SSIM_PSNR_project_defined")},
         "model": "baseline_rf",
         "n_features": len(str(base["baseline_rf"]["features"]).split(","))},
        neural_row("hybrid_all25", met_c, "C_cnn_plus_25"),
        neural_row("hybrid_final",
                   load_metrics("hybrid_final"), "D_cnn_plus_selected"),
    ]
    out = COMPARISON_RESULTS_DIR / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")
    print(pd.DataFrame(rows)[["ablation", "ssim_r2", "psnr_r2",
                              "avg_R2_SSIM_PSNR_project_defined"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
