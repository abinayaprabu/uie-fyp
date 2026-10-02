# Evaluation — sealed test protocol, metrics, and what may be claimed

*Written 2026-10-02. This document describes the evaluation that is actually
run by `scripts/evaluate_hybrid.py`; it does not add any claim.*

## 1. Protocol

- **Split:** the frozen 133-image test split (seed 42). It is read exactly once,
  after every model- and hyper-parameter decision has been made on
  train/validation.
- **Systems compared** (all on the same 133 images, same references, same
  metric code):

  | Row | System | Image source |
  |---|---|---|
  | 1 | Raw | untouched input, only the frozen width-600 resize |
  | 2 | Classical | the frozen preprocessing pipeline |
  | 3 | Existing U-Net | the prior-work baseline (`dataset/enhanced-test/`) |
  | 4 | Image-only (ours) | `results/enhancement/enh224_imgonly/enhanced/` |
  | 5 | Feature-guided (ours) | `results/enhancement/enh224_featguided/enhanced/` |

- **Resolution handling:** the existing U-Net was trained/evaluated at full
  resolution; the proposed model works at 224 and its output is un-letterboxed
  and resized back. That resampling is a disclosed handicap of the proposed
  model (identical for rows 4 and 5, so the ablation between them is fair).

## 2. Metrics

Full detail in `docs/metrics_explanation.md`. In short: PSNR and SSIM against
the UIEB reference; UIQM (Panetta) and UCIQE (Yang & Sowmya) without a
reference, computed by `src/nr_metrics.py` with every convention written in the
module. **UIQM/UCIQE are never treated as distance to the ground truth**, and
their absolute values are never compared with other papers' tables.

## 3. Two gates before any number is reported

1. **Metric-definition gate.** The evaluation recomputes the classical
   pipeline's SSIM/PSNR for all 133 images and compares them with the committed
   label file. Observed maximum deviation: 1.1e-16 (SSIM) / 3.6e-15 (PSNR).
   Any larger deviation means the metric definition drifted and the run is not
   reported.
2. **Provenance.** Every evaluated image is read back from disk (PNG written by
   the enhancement step), never from an in-memory tensor of an earlier stage,
   so a stale checkpoint cannot silently produce a good-looking number.

## 4. Statistics on the comparison

For each pair of systems and each metric, over the 133 paired images:

- mean/median/standard deviation of the per-image difference,
- wins/losses/ties,
- 95 % percentile-bootstrap CI on the mean difference (4000 resamples),
- Wilcoxon signed-rank test on the per-image losses.

The required ablation is **feature-guided − image-only**; the sign of the
result is reported as measured. "Better", "similar" or "worse" is decided by
that interval, not by a target.

## 5. Qualitative analysis

`results/metrics/comparison_images/` — panels RAW | CLASSICAL | EXISTING
U-NET | IMAGE-ONLY | FEATURE-GUIDED | REFERENCE, each titled with that image's
PSNR/SSIM/UIQM/UCIQE. Cases are selected by a documented, model-independent
rule after the model is frozen: highest guided SSIM (best), median guided SSIM,
largest guided-vs-classical SSIM drop (failure case). Nothing about the model
or its training depends on which cases are drawn.

## 6. Output files

| File | Content |
|---|---|
| `results/metrics/final_results.csv` / `.md` | the final table Method × PSNR/SSIM/UIQM/UCIQE with CIs |
| `results/metrics/enhancement_metrics.csv` | per-image metrics for every system |
| `results/metrics/paired_comparisons.csv` | every paired comparison with CI, win/loss and p |
| `results/metrics/comparison_images/*.png` | qualitative panels |
| `results/enhancement/<run>/test_metrics.json` | per-run aggregate, next to its history |

The committed U-Net results (`results/enhancement/unet_128*`, `plots/enhancement_*.png`,
`dataset/enhanced-test/`) are **never overwritten** by this evaluation.

## 7. Claim policy

Allowed: "on the sealed 133-image test split, system X measured PSNR a / SSIM b
/ UIQM c / UCIQE d; the paired difference X − Y was +Δ with 95 % CI [·, ·]".

Not allowed: any claim that the feature-guided model is better before the
ablation interval is read; "state of the art"; cross-paper metric comparisons;
"UIQM proves it looks better"; any number that cannot be re-derived with the
commands in `docs/training.md`.
