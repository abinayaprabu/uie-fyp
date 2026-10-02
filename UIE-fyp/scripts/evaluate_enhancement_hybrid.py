#!/usr/bin/env python3
"""Final evaluation of the feature-guided enhancer on the SEALED TEST SPLIT.

Scores every system that exists for the final comparison table against the SAME
aligned UIEB reference, on the SAME 133 test images, with the SAME metric
definitions:

    raw          untouched input, only the frozen width-600 resize
    classical    the frozen preprocessing pipeline (existing baseline)
    unet         the existing enhancement U-Net (prior work)
    image_only   the proposed architecture with the feature branch OFF
    guided       the proposed architecture with the statistically selected
                 handcrafted features conditioning the bottleneck

Why this script exists separately from ``evaluate_enhancement.py``
------------------------------------------------------------------
The committed U-Net run is sealed evidence: its numbers may be re-derived but
never overwritten. This script writes NEW files only (``results/metrics/``,
``results/enhancement/<run_tag>/``) and reproduces the committed classical
SSIM/PSNR as an equality gate, so a wrong metric definition cannot slip in.

It also computes the two NO-REFERENCE metrics (UIQM, UCIQE) for every system
using ``src/nr_metrics.py`` and states in the output that they measure
colourfulness/sharpness/contrast on the image itself, NOT distance to the
ground truth.

The required ablation (image-only vs feature-guided) is reported as a paired
comparison with percentile-bootstrap CIs and a Wilcoxon signed-rank test over
the 133 images -- both directions are printed, win or lose.

Usage:
    python scripts/evaluate_enhancement_hybrid.py --allow-missing
    python scripts/evaluate_enhancement_hybrid.py            # needs both runs
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
    DATASET_DIR, ENHANCED_DIR, ENHANCEMENT_RESULTS_DIR, FEATURE_RESULTS_DIR,
    HYBRID_RUN_GUIDED, HYBRID_RUN_IMAGE_ONLY, PREPROCESSED_DIR, RAW_DIR,
    RESULTS_DIR,
)
from src.iqa import compute_ssim_psnr  # noqa: E402
from src.metrics import (  # noqa: E402
    bootstrap_mean_ci, bootstrap_mean_delta_ci, paired_wilcoxon,
)
from src.nr_metrics import uciqe, uiqm  # noqa: E402
from src.preprocess import resize_to_width  # noqa: E402

QUALITY_CSV = FEATURE_RESULTS_DIR / "feature_quality_dataset.csv"
METRICS_DIR = RESULTS_DIR / "metrics"
CASES_DIR = METRICS_DIR / "comparison_images"

# system -> how to find its image for a test name (or None when missing)
SYSTEM_ORDER = ("raw", "classical", "unet", "image_only", "guided")
DISPLAY = {
    "raw": "Raw", "classical": "Classical", "unet": "Existing U-Net",
    "image_only": "Image-only (ours)", "guided": "Feature-guided (ours)",
}


def system_path(system: str, name: str, hybrid_dir: Path | None = None) -> Path | None:
    if system == "raw":
        return DATASET_DIR / "raw-890" / name
    if system == "classical":
        return PREPROCESSED_DIR / name
    if system == "unet":
        return ENHANCED_DIR / name
    if system == "image_only":
        return ENHANCEMENT_RESULTS_DIR / HYBRID_RUN_IMAGE_ONLY / "enhanced" / name
    if system == "guided":
        return ENHANCEMENT_RESULTS_DIR / HYBRID_RUN_GUIDED / "enhanced" / name
    raise ValueError(system)


def load_system_image(system: str, name: str) -> np.ndarray:
    """Read one system's image at the reference geometry (BGR uint8)."""
    p = system_path(system, name)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"missing {system} image: {p}")
    if system == "raw":
        img = resize_to_width(img)          # identical frozen resize as labels
    return img


def agg(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    lo, hi = bootstrap_mean_ci(a)
    return {"mean": round(float(a.mean()), 6), "median": round(float(np.median(a)), 6),
            "std": round(float(a.std(ddof=1)), 6), "min": round(float(a.min()), 6),
            "max": round(float(a.max()), 6), "n": int(len(a)),
            "mean_ci95": [round(lo, 6), round(hi, 6)]}


def paired_report(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str,
                  lower_is_better_metric: bool = True) -> dict:
    """Paired comparison of a vs b over the same images (higher score = better).

    ``lower_is_better_metric`` only affects the Wilcoxon call, which takes the
    per-image LOSS; the deltas are always a - b with the raw score.
    """
    d = a - b
    loss_a = -a if lower_is_better_metric else a
    loss_b = -b if lower_is_better_metric else b
    p = paired_wilcoxon(loss_a, loss_b)
    return {
        "comparison": f"{label_a} - {label_b}",
        "mean_delta": round(float(d.mean()), 6),
        "median_delta": round(float(np.median(d)), 6),
        "delta_std": round(float(d.std(ddof=1)), 6),
        "win": int((d > 1e-9).sum()), "loss": int((d < -1e-9).sum()),
        "tie": int((np.abs(d) <= 1e-9).sum()),
        "wilcoxon_p": round(float(p), 6),
        "delta_ci95": [round(x, 6) for x in bootstrap_mean_delta_ci(d)],
    }


def make_panel(name: str, ref: np.ndarray, images: dict[str, np.ndarray],
               scores: dict[str, dict], out_path: Path, note: str) -> None:
    """RAW | CLASSICAL | U-NET | IMAGE-ONLY | GUIDED | REFERENCE side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems = [s for s in SYSTEM_ORDER if s in images]
    fig, axes = plt.subplots(1, len(systems) + 1, figsize=(3.1 * (len(systems) + 1), 3.6))
    for ax, s in zip(axes, systems):
        ax.imshow(cv2.cvtColor(images[s], cv2.COLOR_BGR2RGB))
        sc = scores[s]
        ax.set_title(f"{DISPLAY[s]}\nPSNR {sc['psnr']:.2f}  SSIM {sc['ssim']:.3f}\n"
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
    ap.add_argument("--allow-missing", action="store_true",
                    help="evaluate the systems whose images exist, mark the rest N/A")
    ap.add_argument("--limit", type=int, default=None, help="first N test images (debug)")
    ap.add_argument("--cases", type=int, default=9,
                    help="number of qualitative panels to draw")
    args = ap.parse_args()

    test_ids = split_ids("test")
    if args.limit:
        test_ids = test_ids[:args.limit]
    assert len(test_ids) in (133, args.limit or 133), "unexpected test split size"

    # Which systems are actually available?
    available = []
    for s in SYSTEM_ORDER:
        missing = [n for n in test_ids if system_path(s, n) is None
                   or not system_path(s, n).exists()]
        if missing:
            if args.allow_missing:
                print(f"[skip] {s}: {len(missing)}/{len(test_ids)} images missing "
                      f"(e.g. {missing[0]})")
            else:
                raise FileNotFoundError(
                    f"{s}: {len(missing)} images missing, e.g. {missing[0]}\n"
                    f"  build them first, or pass --allow-missing")
        else:
            available.append(s)
    if not available:
        raise SystemExit("no system available")
    print(f"Evaluating: {', '.join(available)}\n")

    committed = pd.read_csv(QUALITY_CSV).set_index("image_name")
    t0 = time.time()
    rows = []
    max_dev = {"ssim": 0.0, "psnr": 0.0}
    for i, name in enumerate(test_ids, 1):
        ref = aligned_reference(name)
        images, scores = {}, {}
        for s in available:
            img = load_system_image(s, name)
            if img.shape != ref.shape:
                raise ValueError(f"{name}/{s}: {img.shape} != reference {ref.shape}")
            images[s] = img
            ssim, psnr = compute_ssim_psnr(img, ref)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            scores[s] = {"ssim": ssim, "psnr": psnr,
                         "uiqm": uiqm(rgb), "uciqe": uciqe(rgb)}
        # Equality gate: the classical numbers must reproduce the label file.
        if "classical" in scores:
            max_dev["ssim"] = max(max_dev["ssim"],
                                  abs(scores["classical"]["ssim"] - float(committed.loc[name, "ssim"])))
            max_dev["psnr"] = max(max_dev["psnr"],
                                  abs(scores["classical"]["psnr"] - float(committed.loc[name, "psnr"])))
        row = {"image_name": name, "height": ref.shape[0], "width": ref.shape[1]}
        for s in available:
            for m in ("ssim", "psnr", "uiqm", "uciqe"):
                row[f"{m}_{s}"] = scores[s][m]
        rows.append(row)
        if i % 20 == 0 or i == len(test_ids):
            print(f"  {i}/{len(test_ids)}  ({(time.time()-t0)/i:.2f} s/image)", flush=True)

    df = pd.DataFrame(rows)
    print(f"\nclassical vs committed labels: max |dSSIM| {max_dev['ssim']:.2e}, "
          f"max |dPSNR| {max_dev['psnr']:.2e}")
    matches = max_dev["ssim"] < 1e-9 and max_dev["psnr"] < 1e-6
    print(f"  -> metric definition reproduces feature_quality_dataset.csv: "
          f"{'YES' if matches else 'NO — INVESTIGATE'}")

    # ---------------- aggregate table -------------------------------------
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(METRICS_DIR / "enhancement_metrics.csv", index=False)

    table_rows = []
    for s in available:
        rec = {"method": DISPLAY[s], "system": s, "n": int(len(df))}
        for m in ("psnr", "ssim", "uiqm", "uciqe"):
            a = agg(df[f"{m}_{s}"].tolist())
            rec[f"{m}_mean"] = a["mean"]
            rec[f"{m}_median"] = a["median"]
            rec[f"{m}_ci95_lo"], rec[f"{m}_ci95_hi"] = a["mean_ci95"]
        table_rows.append(rec)
    final = pd.DataFrame(table_rows)
    final.to_csv(METRICS_DIR / "final_results.csv", index=False)

    # ---------------- paired comparisons ----------------------------------
    pairs = []
    if "guided" in available and "image_only" in available:
        pairs.append(("guided", "image_only", "REQUIRED ablation: features vs no features"))
    if "guided" in available and "unet" in available:
        pairs.append(("guided", "unet", "proposed vs prior U-Net baseline"))
    if "image_only" in available and "unet" in available:
        pairs.append(("image_only", "unet", "proposed architecture (no features) vs U-Net"))
    if "classical" in available and "raw" in available:
        pairs.append(("classical", "raw", "existing preprocessing vs untouched input"))

    comp_rows = []
    for a, b, note in pairs:
        for m in ("ssim", "psnr", "uiqm", "uciqe"):
            r = paired_report(df[f"{m}_{a}"].to_numpy(), df[f"{m}_{b}"].to_numpy(), a, b)
            r["metric"] = m
            r["note"] = note
            comp_rows.append(r)
    if comp_rows:
        pd.DataFrame(comp_rows).to_csv(METRICS_DIR / "paired_comparisons.csv", index=False)

    # ---------------- markdown version of the final table -----------------
    lines = ["# Final results — sealed test split (n=%d)\n" % len(df),
             "| Method | PSNR (dB) | SSIM | UIQM | UCIQE |", "|---|---|---|---|---|"]
    for _, r in final.iterrows():
        lines.append(f"| {r['method']} | {r['psnr_mean']:.4f} | {r['ssim_mean']:.4f} | "
                     f"{r['uiqm_mean']:.4f} | {r['uciqe_mean']:.4f} |")
    lines += ["", "Means over the same 133 images; 95% percentile-bootstrap CIs on the "
                  "mean are in `final_results.csv`.", "",
              "UIQM and UCIQE are NO-REFERENCE scores computed by `src/nr_metrics.py`: "
              "higher means more colour/sharpness/contrast in the image itself, NOT "
              "closer to the reference. They must never be compared against numbers "
              "from other papers (different conventions), only between the rows above."]
    (METRICS_DIR / "final_results.md").write_text("\n".join(lines) + "\n")

    # ---------------- qualitative panels ----------------------------------
    if args.cases and "guided" in available:
        CASES_DIR.mkdir(parents=True, exist_ok=True)
        # Selection rule (documented, test-set-independent of training):
        #   best    = highest guided SSIM,  median = median guided SSIM,
        #   failure = largest drop of guided SSIM vs the classical pipeline.
        s = df["ssim_guided"]
        idx_best, idx_med = s.idxmax(), (s - s.median()).abs().idxmin()
        idx_fail = (df["ssim_guided"] - df["ssim_classical"]).idxmin()
        chosen = [("best_guided", idx_best), ("median_guided", idx_med),
                  ("failure_case", idx_fail)]
        if args.cases > 3:   # fill with the strongest feature-guided wins
            order = (df["ssim_guided"] - df["ssim_classical"]).sort_values(ascending=False)
            for i in order.index:
                if len(chosen) >= args.cases:
                    break
                if i not in [c[1] for c in chosen]:
                    chosen.append((f"guided_win_{len(chosen)-2}", i))
        for tag, idx in chosen:
            name = df.loc[idx, "image_name"]
            ref = aligned_reference(name)
            imgs = {sym: load_system_image(sym, name) for sym in available}
            sc = {sym: {"ssim": float(df.loc[idx, f"ssim_{sym}"]),
                        "psnr": float(df.loc[idx, f"psnr_{sym}"]),
                        "uiqm": float(df.loc[idx, f"uiqm_{sym}"]),
                        "uciqe": float(df.loc[idx, f"uciqe_{sym}"])} for sym in available}
            out = CASES_DIR / f"{tag}_{name}"
            make_panel(name, ref, imgs, sc, out, tag.replace("_", " "))
            print(f"  panel: {out}.png")

    # ---------------- console summary -------------------------------------
    print("\n" + "=" * 84)
    print(f"SEALED TEST SET (n={len(df)}) — means")
    print("=" * 84)
    print(f"{'method':<24}{'PSNR':>9}{'SSIM':>9}{'UIQM':>9}{'UCIQE':>9}")
    for _, r in final.iterrows():
        print(f"{r['method']:<24}{r['psnr_mean']:>9.4f}{r['ssim_mean']:>9.4f}"
              f"{r['uiqm_mean']:>9.4f}{r['uciqe_mean']:>9.4f}")
    for r in comp_rows:
        if r["metric"] in ("ssim", "psnr"):
            print(f"  {r['comparison']:<26} {r['metric']}: {r['mean_delta']:+.4f} "
                  f"CI [{r['delta_ci95'][0]:+.4f}, {r['delta_ci95'][1]:+.4f}] "
                  f"win {r['win']}/loss {r['loss']} p={r['wilcoxon_p']:.4f}")

    # per-run metrics.json for the two hybrid runs (kept next to their history)
    for sym, tag in (("image_only", HYBRID_RUN_IMAGE_ONLY), ("guided", HYBRID_RUN_GUIDED)):
        if sym in available:
            run_dir = ENHANCEMENT_RESULTS_DIR / tag
            payload = {
                "run": tag, "variant": sym, "split": "test", "n_test": int(len(df)),
                "metric_definition_matches_committed_labels": bool(matches),
                "max_deviation_vs_committed": {"ssim": float(max_dev["ssim"]),
                                               "psnr": float(max_dev["psnr"])},
            }
            for m in ("psnr", "ssim", "uiqm", "uciqe"):
                payload[m] = agg(df[f"{m}_{sym}"].tolist())
            for r in comp_rows:
                if r["comparison"].startswith(sym) and r["metric"] in ("ssim", "psnr"):
                    payload[f"{r['metric']}_vs_{r['comparison'].split(' - ')[1]}"] = r
            with open(run_dir / "test_metrics.json", "w") as f:
                json.dump(payload, f, indent=2)
    print(f"\nWrote {METRICS_DIR/'final_results.csv'}, enhancement_metrics.csv, "
          f"paired_comparisons.csv, final_results.md")
    if args.cases:
        print(f"Wrote qualitative panels to {CASES_DIR}/")
    print(f"Elapsed {time.time()-t0:.1f} s")
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
