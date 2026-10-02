# Viva notes — questions, answers, and the exact wording to use

*Written 2026-10-02 for the final feature-guided enhancement build. Every
answer matches the code in this repository; numbers marked "measured" can be
re-derived with the commands in `docs/training.md` and `docs/evaluation.md`.

This file also serves as the `viva_questions.md` required earlier: it contains
the 30-question list, the required conceptual answer (section 2), and the
one-line answers to the "why" list of the specification (section 3).*

---

## 1. The 30 questions

**Q1. What is the project in one sentence?**
A CNN reconstructs an enhanced underwater image from the image itself, guided
by a small set of handcrafted quality features that were chosen by a
train-only statistical procedure.

**Q2. Why does the image go into the CNN if the features already exist?**
"Handcrafted features provide compact and interpretable information about the
image's colour, contrast, texture and sharpness characteristics. However, they
are mostly global scalar descriptors and do not preserve the spatial
arrangement of image structures. Since underwater image enhancement is a
spatial reconstruction task, the original image must be processed by a CNN to
learn spatial feature maps. The selected handcrafted features are therefore
used as additional guidance to condition the CNN representation rather than
replacing the image input." *(this exact answer is required by the brief)*

**Q3. What are the 25 features?** Six statistical (`mean, std, variance,
entropy, dynamic_range, rms_contrast`), seven colour (`mean_red, mean_green,
mean_blue, colorfulness, red_ratio, mean_saturation, mean_value`), eight
GLCM/texture (`contrast, dissimilarity, correlation, homogeneity, energy, ASM,
glcm_entropy, glcm_variance`) and four edge/sharpness (`edge_density,
gradient, laplacian_variance, keypoint_density`).

**Q4. Where are the features computed from?** The full-resolution preprocessed
underwater image (`src/features.py`, called by `scripts/build_feature_dataset.py`).
Never from the reference image — that would be target leakage (verified in
`scripts/leakage_audit.py`, check 5).

**Q5. What statistical test selects the features and why?** Univariate
regression F-test (`f_regression`), because the targets are continuous (SSIM
and PSNR). H0: the regression slope of the target on that single feature is
zero. A t-test is not used for this — it would only fit a two-group
comparison.

**Q6. What is an effect size here, and why is Pearson r not a second test?**
Pearson r is the correlation implied by the same linear model as the F-test, so
it is reported as the *size* of the effect, not as independent evidence.
Spearman ρ is added as a monotonicity robustness check.

**Q7. Why multiple-testing correction?** 25 features × 2 targets = 50 tests;
at α = 0.05 about 2.5 false positives would be expected by chance. BH-FDR
controls the expected false-discovery proportion while keeping more power than
Bonferroni; both are reported (`p_raw`, `p_bh_fdr`, `p_bonferroni`).

**Q8. What does the Random Forest add over the F-test?** The F-test detects
*linear* association; permutation importance measures how much a flexible
model actually relies on a feature when predicting the target on **validation**
data. The RF is fitted on train only, never on test.

**Q9. How are the two targets combined?** Each target ranks the features
separately; the combined rank is the mean of the two 1-based ranks
(`stats/selector.py::combine_rankings`), ties broken by mean normalised
importance and then alphabetically.

**Q10. Why Pearson redundancy filtering, and why 0.90?** Two features with
|r| ≥ 0.90 carry almost the same information, so keeping both adds noise and
dimension without information. The threshold is a documented, conventional
choice; the higher-ranked feature survives. This is feature-feature
redundancy, not a target-correlation test.

**Q11. Why are some pairs mathematically redundant?** `variance = std²` and
`ASM = energy²` are algebraic identities; they are recorded in
`results/statistics/algebraically_linked_pairs.csv`.

**Q12. How many features survived, and why not 14?** 10 — the statistically
justified set. The specification explicitly says not to force 14. The old
14-feature list is preserved at `results/feature/final_selected_features.csv`
and both are compared in `results/feature_selection/feature_selection_report.md`.

**Q13. What is the architecture?** Encoder 3→32→64→128→256 with
MaxPool2d between blocks (spatial 224→112→56→28→14); the features go through
`Linear(k→32) → ReLU → Dropout(0.2) → Linear(32→2·256)` producing γ and β that
modulate the 256×14×14 bottleneck (FiLM, identity at initialisation);
decoder 256→128→64→32→16 with skip connections from 128×28×28, 64×56×56 and
32×112×112, output 3×224×224 through a sigmoid.

**Q14. Why FiLM and not concatenation?** FiLM changes the *representation*
channel-wise at the bottleneck, so the guidance can scale or suppress learned
spatial responses. Concatenation would append information that the decoder
could ignore; the sanity test shows the FiLM path actually changes the output.

**Q15. What is the parameter count?** 1,188,211 with the feature branch,
1,170,963 image-only (the branch itself is 17,248 parameters); the existing
U-Net baseline has 472,259. Measured by `scripts/test_hybrid_shapes.py`.

**Q16. What does "256" mean?** The number of channels / learned feature
dimension at the bottleneck (256×14×14), not a resolution. GLCM also uses 256
grey levels in the feature extractor — an unrelated use of the same number.

**Q17. Why is the input 224×224 when preprocessing is 600 px wide?** 224 is the
CNN input resolution; 600 px is the preprocessing/feature resolution. The image
is letterboxed (aspect-preserving resize + black padding), the output is
un-letterboxed and resized back before scoring. The resampling is disclosed as
a handicap.

**Q18. What loss?** L1, the primary setting; an optional
`L1 + λ·(1 − SSIM)` exists with λ configurable, but λ is never claimed to come
from a paper and the reported runs use pure L1.

**Q19. How is the model selected?** Best checkpoint on the validation split
(134 images) by SSIM; early stopping with patience 12 epochs; the test split is
not read during training. Test is evaluated once, at the end.

**Q20. How is augmentation kept safe?** Only rigid flips, applied to both
members of a pair (Klein four-group). The features were verified unchanged
(the mirror test: 23/25 features exactly invariant, 2 within 6e-5), so the
cached features stay valid; no photometric augmentation.

**Q21. How do you know the features influence the network?** Two checks: at
initialisation FiLM is the identity, so the guided model equals the image-only
model exactly; after a few optimizer steps, zeroing or shuffling the features
changes the output (max |Δ| 1.5e-2 and 3.0e-2 in the unit test) and the
conditioning branch has non-zero gradients.

**Q22. What is the required ablation?** image-only vs feature-guided under
identical settings: same data, aug, loss, epochs, seeds; only the conditioning
branch differs. Reported with paired bootstrap CIs and a Wilcoxon signed-rank
test on the 133 test images.

**Q23. Metrics: what is measured, and what is not?** PSNR and SSIM against the
UIEB reference (full-reference); UIQM and UCIQE without a reference. UIQM/UCIQE
reward colour, sharpness and contrast and can rise while the image moves away
from the ground truth — they are complementary, never proof.

**Q24. Why does UIQM sometimes disagree with PSNR/SSIM?** Because it does not
know the reference. Measured example: the existing U-Net scores higher than the
classical pipeline on PSNR, SSIM and UCIQE but *lower* on UIQM.

**Q25. Is the classical pipeline better than the raw image?** Not
significantly on the sealed test split: ΔSSIM +0.0034 (CI [−0.017, +0.023],
p = 0.43). This is reported honestly rather than hidden.

**Q26. What is the biggest limitation?** The 224-px letterbox input and the
resize back to full resolution handicap the proposed model against the
full-resolution U-Net; the reference is a single image per scene (no
distribution of "correct" enhancements); and the feature set is selected
against the classical pipeline's own quality scores, which is a proxy.

**Q27. What would you do next with more time?** A λ-tuned L1+SSIM run, a second
seed for both variants to report variance, and a masked L1 that ignores the
letterbox bars.

**Q28. What does the leakage audit prove?** Twelve checks: split sizes and
disjointness; statistics reproducible from train rows only; the full selection
funnel recomputed without test; the scaler fitted on train only (checkpoint
check); augmentation train-only; the reference never an input; metrics
deterministic from the PNG bytes on disk.

**Q29. What did the previous project direction conclude, and why was it
replaced?** The four quality-prediction models predicted scalar SSIM/PSNR;
image-only and feature-fused predictors were indistinguishable (R² 0.4659 vs
0.4658), and a feature-only enhancement study showed the features alone cannot
place structures (best case ≈ conditional mean → blur). Those results are the
evidence for the final design and are archived under
`archive/quality_prediction/`.

**Q30. Where does each required artefact live?** Labels and split in
`results/feature/`; statistics in `results/statistics/`; the clean selection
deliverable in `results/feature_selection/`; final metrics in
`results/metrics/`; training histories and checkpoints in
`results/enhancement/<run>/`; panels in `results/metrics/comparison_images/`.

---

## 2. The required conceptual answer (repeat this one verbatim)

> "Handcrafted features provide compact and interpretable information about the
> image's colour, contrast, texture and sharpness characteristics. However,
> they are mostly global scalar descriptors and do not preserve the spatial
> arrangement of image structures. Since underwater image enhancement is a
> spatial reconstruction task, the original image must be processed by a CNN to
> learn spatial feature maps. The selected handcrafted features are therefore
> used as additional guidance to condition the CNN representation rather than
> replacing the image input."

## 3. One-line answers to the "why" list

| Question | Short answer |
|---|---|
| Why does the image enter the CNN? | Enhancement is spatial reconstruction; global scalars cannot place structures. |
| Why extract handcrafted features too? | They encode colour/contrast/texture/sharpness in an interpretable, low-dimensional form that can steer the decoder. |
| What do the 25 features represent? | Global statistics of tone, colour, GLCM texture and edge/sharpness content. |
| Why statistical validation? | To choose features by measured association, not by intuition, and to document the choice. |
| What is `f_regression`? | Univariate linear-model F-test per feature against a continuous target. |
| Why FDR correction? | 50 tests at α = 0.05 would produce ~2.5 false positives by chance. |
| What is RF permutation importance? | The drop in a fitted forest's validation performance when a feature's values are randomly permuted. |
| Why Pearson redundancy filtering? | |r| ≥ 0.90 features carry duplicate information; keep the stronger-ranked one. |
| Why the 0.90 threshold? | Documented convention for "practically collinear"; stated, not hidden. |
| Why do the features guide the CNN? | They summarise the degradation; conditioning lets the same decoder fall back to a different restoration strength per image. |
| Why is the image still needed? | Only the image branch carries spatial information (where the edges, textures and structures are). |
| CNN maps vs handcrafted features? | Learned spatial tensors (256×14×14) versus 25 interpretable global scalars; different roles, not replacements. |
| Why UIEB paired images? | Supervised pairs (underwater, reference) enable reference-based losses and metrics. |
| Why is the test split untouched? | Any number used to choose a model is not a test number; leakage would invalidate the comparison. |
| PSNR vs SSIM vs UIQM vs UCIQE? | PSNR: pixel error (FR). SSIM: structure/luminance/contrast (FR). UIQM: colour+sharpness+contrast (NR). UCIQE: colour variation + luminance range + saturation (NR). |
