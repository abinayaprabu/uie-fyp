#!/usr/bin/env python3
"""Four-metric final evaluation on the SEALED TEST SPLIT (read once, at the end).

Produces the thesis table required by the specification:

    Method | PSNR | SSIM | UIQM | UCIQE

for every system that exists on disk, all on the same 133 images with the same
metric definitions:

    raw          untouched input, only the frozen width-600 resize
    classical    the frozen preprocessing pipeline (existing baseline)
    unet         the existing enhancement U-Net (prior work)
    <run tags>   the trained feature-guided models, e.g.
                 enh224_imgonly    (ablation: no feature guidance)
                 enh224_featguided (proposed: feature guidance ON)

PSNR/SSIM are full-reference (need the UIEB reference); UIQM/UCIQE are
no-reference (they score the image itself). They are reported side by side and
never mixed into a single "better" claim.

A consistency gate re-derives the classical labels: if the recomputed SSIM/PSNR
of the classical pipeline does not reproduce
``results/feature/feature_quality_dataset.csv``, the metric definition has
drifted and no number from the run may be reported.

Outputs -> results/metrics/
    final_results.csv / .md    aggregates with 95% bootstrap CIs
    enhancement_metrics.csv    every image x every system x every metric
    paired_comparisons.csv     paired deltas, win/loss, Wilcoxon p
    comparison_images/*.png    qualitative panels (best / median / failure)
results/enhancement/<run>/test_metrics.json   per-run aggregate

Usage:
    python scripts/evaluate_hybrid.py --limit 4        # quick check
    python scripts/evaluate_hybrid.py                  # full 133 images
    python scripts/evaluate_hybrid.py --cases 0        # skip the panels
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn.dataset_pairs import aligned_reference, split_ids  # noqa: E402
from src.config import (  # noqa: E402
    ENHANCED_DIR, ENHANCEMENT_RESULTS_DIR, FEATURE_RESULTS_DIR, PREPROCESSED_DIR,
    RAW_DIR, RESULTS_DIR,
)
from src.iqa import compute_ssim_psnr  # noqa: E402
from src.metrics import (  # noqa: E402
    bootstrap_mean_ci, bootstrap_mean_delta_ci, paired_wilcoxon,
)
from src.nr_metrics import uciqe, uiqm  # noqa: E402
from src.preprocess import resize_to_width  # noqa: E402

OUT_DIR = RESULTS_DIR / "metrics"
CASES_DIR = OUT_DIR / "comparison_images"
QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"
METRICS = ("psnr", "ssim", "uiqm", "uciqe")


def agg(vals: np.ndarray) -> dict:
    lo, hi = bootstrap_mean_ci(vals)
    return {"mean": round(float(vals.mean()), 6),
            "std": round(float(vals.std(ddof=1)), 6),
            "median": round(float(np.median(vals)), 6),
            "min": round(float(vals.min()), 6), "max": round(float(vals.max()), 6),
            "n": int(vals.size), "ci95_mean": [round(lo, 6), round(hi, 6)]}


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired comparison a vs b (higher score = better)."""
    d = a - b
    p = paired_wilcoxon(-a, -b)          # works on per-image LOSSES
    lo, hi = bootstrap_mean_delta_ci(d)
    return {"mean_delta": round(float(d.mean()), 6),
            "median_delta": round(float(np.median(d)), 6),
            "std_delta": round(float(d.std(ddof=1)), 6),
            "ci95_mean_delta_lo": round(lo, 6), "ci95_mean_delta_hi": round(hi, 6),
            "win": int((d > 1e-9).sum()), "loss": int((d < -1e-9).sum()),
            "tie": int((np.abs(d) <= 1e-9).sum()),
            "wilcoxon_p": round(float(p), 6)}


def run_dirs() -> list[str]:
    """Feature-guided run tags present on disk (image-only first, then guided)."""
    tags = []
    for d in sorted(ENHANCEMENT_RESULTS_DIR.iterdir()):
        if (d / "enhanced").is_dir() and any((d / "enhanced").glob("*.png")):
            tags.append(d.name)
    return tags


def make_panel(name: str, images: dict[str, np.ndarray], scores: dict[str, dict],
               labels: dict[str, str], ref: np.ndarray, out_path: Path,
               note: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(images)
    fig, axes = plt.subplots(1, len(order) + 1, figsize=(3.1 * (len(order) + 1), 3.6))
    for ax, key in zip(axes, order):
        ax.imshow(cv2.cvtColor(images[key], cv2.COLOR_BGR2RGB))
        sc = scores[key]
        ax.set_title(f"{labels[key]}\nPSNR {sc['psnr']:.2f}  SSIM {sc['ssim']:.3f}\n"
                     f"UIQM {sc['uiqm']:.2f}  UCIQE {sc['uciqe']:.1f}", fontsize=8)
        ax.axis("off")
    axes[-1].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    axes[-1].set_title("Reference (UIEB)", fontsize=8)
    axes[-1].axis("off")
    fig.suptitle(f"{name} — {note}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-unet", action="store_true",
                    help="do not include the prior-work U-Net row")
    ap.add_argument("--cases", type=int, default=9,
                    help="qualitative panels to draw (0 = none)")
    ap.add_argument("--panel-run", default=None,
                    help="run tag used to choose the panel cases (default: "
                         "the tag containing 'featguided', else the last run)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_ids = split_ids("test")
    if args.limit:
        test_ids = test_ids[:args.limit]
    elif len(test_ids) != 133:
        raise ValueError(f"expected 133 test images, got {len(test_ids)}")

    # system key -> (label, loader)
    systems: dict[str, tuple[str, object]] = {
        "raw": ("Raw", lambda n: resize_to_width(
            cv2.imread(str(RAW_DIR / n), cv2.IMREAD_COLOR))),
        "classical": ("Classical", lambda n: cv2.imread(
            str(PREPROCESSED_DIR / n), cv2.IMREAD_COLOR)),
    }
    if not args.skip_unet and (ENHANCED_DIR / test_ids[0]).exists():
        systems["unet"] = ("Existing U-Net", lambda n: cv2.imread(
            str(ENHANCED_DIR / n), cv2.IMREAD_COLOR))
    tags = run_dirs()
    for tag in tags:
        systems[tag] = (tag, (lambda t: lambda n: cv2.imread(
            str(ENHANCEMENT_RESULTS_DIR / t / "enhanced" / n),
            cv2.IMREAD_COLOR))(tag))
    if not tags:
        print("NOTE: no trained feature-guided run found under "
              f"{ENHANCEMENT_RESULTS_DIR}/*/enhanced/ -- the table will contain "
              "only raw / classical / unet.")

    print(f"Evaluating {len(test_ids)} test images on systems: {list(systems)}")
    committed = pd.read_csv(QUALITY_CSV).set_index("image_name")

    rows, max_dev = [], {"ssim": 0.0, "psnr": 0.0}
    t0 = time.time()
    for i, name in enumerate(test_ids, 1):
        ref = aligned_reference(name)
        row = {"image_name": name, "height": ref.shape[0], "width": ref.shape[1]}
        for key, (_, loader) in systems.items():
            img = loader(name)
            if img is None:
                raise FileNotFoundError(f"{key}: could not read {name}")
            if img.shape != ref.shape:
                raise ValueError(f"{key}/{name}: {img.shape} != ref {ref.shape}")
            ssim, psnr = compute_ssim_psnr(img, ref)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            row[f"psnr__{key}"] = psnr
            row[f"ssim__{key}"] = ssim
            row[f"uiqm__{key}"] = uiqm(rgb)
            row[f"uciqe__{key}"] = uciqe(rgb)
        if "classical" in systems:
            max_dev["ssim"] = max(max_dev["ssim"], abs(
                row["ssim__classical"] - float(committed.loc[name, "ssim"])))
            max_dev["psnr"] = max(max_dev["psnr"], abs(
                row["psnr__classical"] - float(committed.loc[name, "psnr"])))
        rows.append(row)
        if i % 25 == 0 or i == len(test_ids):
            print(f"  {i}/{len(test_ids)}  ({(time.time()-t0)/i:.2f} s/image)",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "enhancement_metrics.csv", index=False)

    gate = max_dev["ssim"] < 1e-9 and max_dev["psnr"] < 1e-6
    print(f"consistency gate: max |dSSIM| {max_dev['ssim']:.2e}, "
          f"max |dPSNR| {max_dev['psnr']:.2e} -> "
          f"{'PASS (same metric definition as the labels)' if gate else 'FAIL'}")

    # ---- aggregate table --------------------------------------------------
    table = []
    for key, (label, _) in systems.items():
        rec = {"method": label, "system": key, "n": int(len(df))}
        for m in METRICS:
            a = agg(df[f"{m}__{key}"].to_numpy())
            rec[f"{m}_mean"] = a["mean"]
            rec[f"{m}_std"] = a["std"]
            rec[f"{m}_ci95_lo"], rec[f"{m}_ci95_hi"] = a["ci95_mean"]
        table.append(rec)
    table_df = pd.DataFrame(table).sort_values("psnr_mean", ascending=False).reset_index(drop=True)
    table_df.to_csv(OUT_DIR / "final_results.csv", index=False)

    lines = ["# Final results — sealed test split (n = %d)\n" % len(df),
             "| Method | PSNR (dB) | SSIM | UIQM | UCIQE |", "|---|---|---|---|---|"]
    for _, r in table_df.iterrows():
        lines.append(f"| {r['method']} | {r['psnr_mean']:.4f} ± {r['psnr_std']:.4f} "
                     f"| {r['ssim_mean']:.4f} ± {r['ssim_std']:.4f} "
                     f"| {r['uiqm_mean']:.4f} ± {r['uiqm_std']:.4f} "
                     f"| {r['uciqe_mean']:.4f} ± {r['uciqe_std']:.4f} |")
    lines += ["", "Means ± standard deviations over the same 133 images; 95% "
                  "percentile-bootstrap CIs on the mean are in `final_results.csv`.", "",
              "PSNR/SSIM are full-reference (distance to the UIEB reference). "
              "UIQM/UCIQE are no-reference: they reward colour/sharpness/contrast in "
              "the image itself, are NOT a distance to the ground truth, and are never "
              "compared with values from other papers."]
    (OUT_DIR / "final_results.md").write_text("\n".join(lines) + "\n")

    # ---- paired comparisons ----------------------------------------------
    comp_rows = []
    keys = list(systems)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            for m in METRICS:
                comp_rows.append({"metric": m, "a": a, "b": b,
                                  **paired(df[f"{m}__{a}"].to_numpy(),
                                           df[f"{m}__{b}"].to_numpy())})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUT_DIR / "paired_comparisons.csv", index=False)

    # ---- per-run metrics next to their histories --------------------------
    for key, (label, _) in systems.items():
        if key in ("raw", "classical", "unet"):
            continue
        payload = {"run": key, "split": "test", "n_test": int(len(df)),
                   "metric_definition_matches_committed_labels": bool(gate),
                   "max_deviation_vs_committed": {"ssim": float(max_dev["ssim"]),
                                                  "psnr": float(max_dev["psnr"])}}
        for m in METRICS:
            payload[m] = agg(df[f"{m}__{key}"].to_numpy())
        payload["paired_vs"] = comp[comp["a"] == key].to_dict(orient="records")
        with open(ENHANCEMENT_RESULTS_DIR / key / "test_metrics.json", "w") as f:
            json.dump(payload, f, indent=2)

    # ---- qualitative panels ----------------------------------------------
    if args.cases and tags:
        panel_tag = args.panel_run or next(
            (t for t in tags if "featguided" in t), tags[-1])
        CASES_DIR.mkdir(parents=True, exist_ok=True)
        s = df[f"ssim__{panel_tag}"]
        idx_best = s.idxmax()
        idx_med = (s - s.median()).abs().idxmin()
        idx_fail = (df[f"ssim__{panel_tag}"] - df["ssim__classical"]).idxmin()
        chosen = [("best_guided", idx_best), ("median_guided", idx_med),
                  ("failure_case", idx_fail)]
        wins = (df[f"ssim__{panel_tag}"] - df["ssim__classical"]).sort_values(
            ascending=False)
        for idx in wins.index:
            if len(chosen) >= args.cases:
                break
            if idx not in [c[1] for c in chosen]:
                chosen.append((f"guided_win_{len(chosen)-2}", idx))
        for tag, idx in chosen:
            name = df.loc[idx, "image_name"]
            ref = aligned_reference(name)
            imgs = {k: systems[k][1](name) for k in systems}
            sc = {k: {m: float(df.loc[idx, f"{m}__{k}"]) for m in METRICS}
                  for k in systems}
            labels = {k: systems[k][0] for k in systems}
            make_panel(name, imgs, sc, labels, ref,
                       CASES_DIR / f"{tag}_{Path(name).stem}", tag.replace("_", " "))
            print(f"  panel: {CASES_DIR / (tag + '_' + Path(name).stem)}.png")

    # ---- console summary --------------------------------------------------
    print("\n" + "=" * 78)
    print(f"FINAL TABLE (sealed test, n={len(df)})")
    print("=" * 78)
    print(f"{'method':<26}{'PSNR':>9}{'SSIM':>9}{'UIQM':>9}{'UCIQE':>9}")
    for _, r in table_df.iterrows():
        print(f"{r['method']:<26}{r['psnr_mean']:>9.4f}{r['ssim_mean']:>9.4f}"
              f"{r['uiqm_mean']:>9.4f}{r['uciqe_mean']:>9.4f}")
    for tag in tags:
        for _, r in comp.iterrows():
            if r["a"] == tag and r["b"] in ("enh224_imgonly", "unet", "classical") \
                    and r["metric"] in ("ssim", "psnr"):
                print(f"  {r['a']} - {r['b']} {r['metric']}: {r['mean_delta']:+.4f} "
                      f"CI [{r['ci95_mean_delta_lo']:+.4f}, {r['ci95_mean_delta_hi']:+.4f}] "
                      f"win {r['win']}/loss {r['loss']} p={r['wilcoxon_p']:.4f}")
    print(f"\nWrote {OUT_DIR}/final_results.csv, enhancement_metrics.csv, "
          f"paired_comparisons.csv, final_results.md")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
