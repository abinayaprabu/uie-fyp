# Metrics used in this project — what each one is, and what it is not

*Written 2026-10-02 for the locked build. Every formula here is the one that is
actually implemented; the file names and line-level conventions are listed so
that a number in the report can always be traced back to code.*

The project reports **four** metrics on the sealed 133-image test split. They
are two families that answer different questions, and the final table keeps
them in separate columns on purpose:

| Family | Metrics | Needs the reference? | Question it answers |
|---|---|---|---|
| Full-reference (FR) | PSNR, SSIM | yes | "how close is the output to the UIEB reference image?" |
| No-reference (NR) | UIQM, UCIQE | no | "does the image itself look colourful, sharp, contrasty?" |

**A no-reference score can go up while the image moves AWAY from the ground
truth.** More colour and contrast is not the same thing as being correct.
That sentence must appear wherever UIQM/UCIQE are quoted.

---

## 1. PSNR — peak signal-to-noise ratio (full-reference)

Implemented in `src/iqa.py::compute_ssim_psnr` (and independently in
`psnr_from_scratch` as a cross-check).

    MSE  = mean over all pixels and channels of (prediction - reference)^2
    PSNR = 10 * log10( 255^2 / MSE )      [dB]

- **What it measures:** average pixel error, in decibels. Higher is better
  (closer to the reference). It is the standard first number in every image
  restoration table.
- **What it does NOT measure:** perceptual quality. A slightly blurred output
  can have a *better* PSNR than a sharper one that is shifted by a pixel, and
  a perfectly plausible restoration that differs from the single UIEB
  reference is punished. That is why SSIM sits next to it.
- **Our conventions:** both images are BGR uint8, converted to RGB;
  `data_range=255`; the **reference is the first argument** everywhere
  (`compute_ssim_psnr(preprocessed_or_enhanced, reference)`), because
  swapping the arguments would silently change SSIM's value.

## 2. SSIM — structural similarity index (full-reference)

`skimage.metrics.structural_similarity(pred, ref, channel_axis=2, data_range=255)`.

    SSIM(x, y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2)
               / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))

with a 7×7 uniform window by default in scikit-image, `C1 = (0.01*255)^2`,
`C2 = (0.03*255)^2`. It compares local means (luminance), local variances
(contrast) and local covariance (structure). Range [-1, 1]; higher is better.

- **Why it matters here:** the project's targets (the 25 handcrafted features
  and the 14/10-feature selection) are ranked against **SSIM and PSNR of the
  classical pipeline**, so these are the metrics the whole selection procedure
  optimizes a proxy for.
- **Cross-check:** `ssim_from_scratch` in the same file re-implements the
  formula; `scripts/verify_enhancement.py` requires the two to agree before
  the committed numbers may be reused.

## 3. UIQM — underwater image quality measure (no-reference)

`src/nr_metrics.py::uiqm` = Panetta, Gao & Agaian (2016):

    UIQM = 0.0282 * UICM + 0.2953 * UISM + 3.5753 * UIConM

- **UICM (colourfulness)** — opponent channels `RG = R - G`,
  `YB = (R+G)/2 - B`, alpha-trimmed statistics (10 % cut from each tail):

      UICM = -0.0268 * sqrt(mu_RG^2 + mu_YB^2)
             + 0.1586 * sqrt(sigma_RG^2 + sigma_YB^2)

- **UISM (sharpness)** — Sobel edge magnitude per RGB channel, then the
  enhancement measure EME over non-overlapping 8×8 blocks:

      EME = mean over blocks of 20*log10(block_max / block_min)
      UISM = 0.299*EME(R) + 0.587*EME(G) + 0.114*EME(B)

- **UIConM (contrast)** — PLIP (parameterised logarithmic image processing)
  version of the same block measure on the mean-intensity image, with the
  standard 8-bit constant 1026:

      c = PLIP_sub(Imax, Imin) / PLIP_add(Imax, Imin)
      UIConM = mean over blocks of c*ln(c)

### Two facts about UIConM that must not be hidden

1. The published term `c*ln(c)` with `c` in (0, 1) is **not monotone**: it is
   ~0 for tiny contrast, reaches -1/e at `c = 1/e`, and rises back toward 0 as
   `c -> 1`. A *less* contrasty image can therefore score a *higher* UIConM.
   `scripts/test_nr_metrics.py` demonstrates this and keeps it visible.
2. Absolute UIQM values depend on implementation choices (block size, the
   gradient floor in EME, PLIP constant, trimming). **Never compare these
   numbers to values printed in other papers.** Compare only between the rows
   of our own final table, all computed by this same file.

Documented conventions in our implementation: 8×8 blocks; block minima in EME
floored at 1 gradient level (the usual guard against log(0), applied
identically to every method); PLIP constant 1026; alpha-trim indices
`int(0.1*K+1) : int(0.9*K)`; UISM weights 0.299/0.587/0.114 (Rec. 601 luma;
some papers print 0.584 for green).

## 4. UCIQE — underwater colour image quality evaluation (no-reference)

`src/nr_metrics.py::uciqe` = Yang & Sowmya (2015):

    UCIQE = 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s

- CIELAB chroma `= sqrt(a*^2 + b*^2)`, `sigma_c` = standard deviation of
  chroma (colour variation),
- `con_l = P99(L*) - P1(L*)` (robust luminance dynamic range, the top and
  bottom 1 % of luminance),
- `mu_s` = mean of per-pixel saturation `chroma / L*` (luminance clipped away
  from zero by a documented epsilon).

The same warning applies: UCIQE rewards colour and contrast, not correctness.

## 5. What is NOT measured

- **No reference-based perceptual metric beyond PSNR/SSIM** (LPIPS, DISTS...)
  is used: they need pretrained networks, which the specification bans.
- **No human study.** The claim "looks better" is supported only by the
  qualitative panels (`results/metrics/comparison_images/`) and by the
  no-reference scores, and it is reported as such.
- **No statistical claim is made about NR metrics unless the paired tests in
  `scripts/evaluate_hybrid.py` actually show it** (wins/losses,
  percentile-bootstrap CI, Wilcoxon signed-rank over the same 133 images).

## 6. Verification of the metric code

- `scripts/test_nr_metrics.py` — flat images score ~0; blurring lowers UISM;
  greyscaling lowers UICM and UCIQE; the fast vectorised EME/UIConM match the
  plain double-loop definitions to ≤ 1e-9; runtime ~0.07 s per full-resolution
  image. All checks pass.
- `scripts/evaluate_hybrid.py` — before reporting anything, it
  recomputes the classical pipeline's SSIM/PSNR for all 133 test images and
  compares them with the committed label file. Observed maximum deviation:
  `1.1e-16` (SSIM) and `3.6e-15` (PSNR), i.e. the metric definition in the
  evaluation script is the same one that produced the project's labels.
- `scripts/verify_enhancement.py` — the same equality gate for the committed
  U-Net numbers.

## 7. Measured values (sealed test split, n = 133)

| Method | PSNR (dB) | SSIM | UIQM | UCIQE |
|---|---|---|---|---|
| Raw (resized only) | 17.0633 | 0.7603 | 7.3322 | 20.4920 |
| Classical pipeline | 17.0889 | 0.7636 | 7.7645 | 22.6355 |
| Existing U-Net | 19.3242 | 0.8003 | 7.6281 | 25.6052 |
| Image-only (ours) | — pending training — | | | |
| Feature-guided (ours) | — pending training — | | | |

The last two rows are written automatically when the twins finish. Note the
honest oddity already visible: the U-Net wins PSNR, SSIM and UCIQE but has a
**lower** UIQM than the classical pipeline — evidence for the warning at the
top of this document.

## 8. Resolution round-trip caveat (must be disclosed in the report)

The proposed model sees a 224×224 letterboxed view and its output is
un-letterboxed and resized back to the reference geometry with INTER_LINEAR
before scoring. The existing U-Net was trained and evaluated at full
resolution on 128-px crops. This resampling is a **handicap of the proposed
model that is disclosed, not hidden**; it applies identically to the image-only
and feature-guided twins, so the required ablation between them is unaffected.
