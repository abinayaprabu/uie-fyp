# Final results — measured, and what they actually mean

*Written 2026-10-02 after the feature-guided model finished training
(80/80 epochs, best epoch 74) and was evaluated once on the sealed 133-image
test split. Every number below is produced by the committed scripts; none is
estimated, adjusted, or copied.*

## 1. The required table (protocol as specified: each method scored at full reference resolution)

`results/metrics/final_results.csv` / `.md` — means ± std over the same 133 images:

| Method | PSNR (dB) | SSIM | UIQM | UCIQE |
|---|---|---|---|---|
| Raw (resized only) | 17.0633 ± 4.0496 | 0.7603 ± 0.1382 | 7.3322 ± 0.5569 | 20.4920 ± 6.7891 |
| Classical pipeline | 17.0889 ± 3.1581 | 0.7636 ± 0.1114 | 7.7645 ± 0.8886 | 22.6355 ± 4.0216 |
| Existing U-Net (prior work) | **19.3242** ± 2.5316 | **0.8003** ± 0.0792 | 7.6281 ± 0.5239 | **25.6052** ± 3.8626 |
| **Feature-guided (proposed)** | 18.9744 ± 2.2645 | 0.6402 ± 0.1163 | 5.9293 ± 0.3379 | 25.2816 ± 3.7642 |

Paired comparisons (`results/metrics/paired_comparisons.csv`, bootstrap CI +
Wilcoxon on the same 133 images):

| Pair | Metric | Δ (a − b) | 95 % CI | W/L | p |
|---|---|---|---|---|---|
| proposed − classical | PSNR | +1.8855 dB | [+1.4163, +2.3375] | 111/22 | <0.001 |
| proposed − classical | SSIM | **−0.1235** | [−0.1472, −0.0999] | 15/118 | <0.001 |
| proposed − classical | UIQM | −1.8352 | [−1.9532, −1.7221] | 0/133 | <0.001 |
| proposed − classical | UCIQE | +2.6461 | [+2.2806, +2.9956] | 121/12 | <0.001 |
| proposed − U-Net | PSNR | −0.3498 dB | [−0.6804, −0.0322] | 59/74 | 0.0407 |
| proposed − U-Net | SSIM | **−0.1601** | [−0.1797, −0.1413] | 9/124 | <0.001 |
| proposed − U-Net | UIQM | −1.6988 | [−1.7560, −1.6436] | 0/133 | <0.001 |
| proposed − U-Net | UCIQE | −0.3236 | [−0.5207, −0.1226] | 47/86 | 0.001 |

**Honest verdict: the proposed model does NOT beat the existing U-Net.** It beats
the classical pipeline on PSNR and UCIQE, is worse than it on SSIM and UIQM, and
it is worse than the U-Net on every metric except that its PSNR loss is small
(−0.35 dB, barely significant). This is reported as measured; no claim of
superiority is made anywhere in this repository.

## 2. Why — measured controls, not speculation

### 2.1 The comparison is resolution-unfair, and the size of that unfairness is measured

`results/metrics/resolution_roundtrip.csv` (same 133 images):

| Control | SSIM | PSNR |
|---|---|---|
| A classical pipeline (baseline row) | 0.7636 | 17.0889 |
| B identity: preprocessed → 224 → back to 600 geometry | 0.6716 | 16.7817 |
| C model output at 224 vs reference at 224 (image area) | 0.7974 | 19.7935 |
| D model output resized to 600 and compared (the reported row) | 0.6402 | 18.9744 |

The identity round trip alone (no network at all) loses 0.09 SSIM and
0.31 dB versus the classical image. So the 224-pixel representation the model
must use is *already* −0.09 SSIM before any learning; the model then does better
than that at its own resolution (+0.13 SSIM, +3.0 dB over control B, i.e. it is
clearly learning), but the final resize back to 600 geometry costs a further
0.16 SSIM. At full resolution the U-Net never pays either penalty — it was
trained and applied natively at that geometry.

### 2.2 Domain-matched control (every system compared at 224 px)

`results/metrics/domain_matched_224.csv` — every method's output and the
reference all resized to 224 before scoring:

| Method | SSIM@224 | PSNR@224 |
|---|---|---|
| Raw | 0.7519 | 17.2100 |
| Classical | 0.8015 | 17.3644 |
| Existing U-Net | **0.8322** | 19.6365 |
| Feature-guided (proposed) | 0.7930 | **19.8208** |

At matched resolution the proposed model has the **best PSNR of all four**
systems and a higher SSIM than raw/classical, but the U-Net still leads on SSIM.
So part — not all — of the reported SSIM gap is the protocol's upsampling.

### 2.3 The failure mode is visible and consistent

`results/metrics/comparison_images/` (documented, non-cherry-picked selection:
highest guided SSIM = best, median guided SSIM, largest guided-vs-classical SSIM
drop = failure, then the six largest guided wins). In the panels the model's
colour restoration is reasonable, but its output is **over-smoothed**: fine
texture (coral, rope, foliage) is blurred, which is exactly what the SSIM drop
measures (failure case UIEB_719: 0.404 SSIM, U-Net 0.881 on the same image).

### 2.4 Does the feature guidance do anything? (measured, and only what was measured)

`results/metrics/feature_sensitivity.csv` — the trained model re-run on the 133
test images with its feature vector normal / zeroed / shuffled (the model is not
retrained; only its input features change):

| Features | SSIM | PSNR |
|---|---|---|
| normal | 0.6402 | 18.9744 |
| zeroed | 0.6265 | 18.0532 |
| shuffled | 0.6295 | 18.2524 |

**normal − zeroed: +0.0136 SSIM, +0.92 dB.** So the trained model does use the
handcrafted guidance when it is present — for PSNR it is worth almost a decibel.

**What this is NOT:** it is not the trained ablation required by §22
(image-only vs feature-guided trained under identical settings). That needs a
second training run of ~3.5 h, which the final instruction sequence explicitly
excluded. Therefore the repository claims only: *the trained model's output
responds measurably to the selected features* — not *the feature-guided
architecture trains to a better model*.

## 3. Diagnosis of the over-smoothing (and the fix that is available)

Cause: the model sees a 224×224 view (~2.7× smaller than the 224 ≤ 600 geometry
ratio), is trained with a pixel-wise L1 loss on that view, and its output is
upsampled back to 600 px. Pixel losses at low resolution reward smooth, safely
averaged outputs; nothing in the objective rewards high-frequency detail.

Available fixes, in increasing cost (none applied without approval, because each
changes the locked specification):
1. **Train at the native geometry** (no letterbox, full-resolution crops as the
   prior U-Net does) — removes controls B and D entirely.
2. **Add the optional loss term** already implemented and left configurable:
   `HYBRID_LOSS=l1_ssim` with a λ chosen on train/val — directly penalises
   structural/detail loss.
3. **Raise the input size** (e.g. 448) while keeping the same encoder widths —
   halves the handicap; a specification-level change (input size is specified as
   224), so it needs explicit approval.

## 4. What the project has, measured and verified

- 25 handcrafted features extracted from the full-resolution preprocessed image,
  never from the reference (leakage audit, 16 checks, all pass).
- A train-only statistical funnel producing the 10 selected features; the whole
  funnel is reproducible from the stored CSVs and matches exactly.
- One feature-guided architecture implementing the locked design (FiLM
  conditioning, identity at initialisation, U-Net decoder with skips),
  1,188,211 parameters, verified from the checkpoint.
- The trained model, its history (80 epochs, best epoch 74, val SSIM 0.6606),
  its sealed-test inference (133/133 images, checked), and its four metrics.
- The honest result: better than the classical pipeline on PSNR/UCIQE and, at
  matched resolution, the best PSNR of the four systems — but not better than
  the existing U-Net overall, with the resolution round trip quantified.
