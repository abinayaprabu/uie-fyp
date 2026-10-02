# Feasibility report — "feature-only" enhancement (14 features → enhanced image)

*Written 2026-10-02. Requested evaluation of: `image → preprocessing → 25 features →
statistical selection → selected features → ONE model → enhanced image → PSNR/SSIM/UIQM/UCIQE`,
where the model receives **only** the feature vector and never the image itself.*

**Evidence discipline used in this report.** Every number is tagged:

* **[MEASURED — this repo, today]** run during this analysis, on committed images;
* **[MEASURED — this repo, earlier]** committed artefacts from earlier verified runs;
* **[THEORY]** an argument from information theory / statistics / architecture, no number claimed;
* **[EXPECTED]** mechanism-based expectation — *not* a result, and never to be quoted as one.

No PSNR/SSIM/UIQM/UCIQE value is predicted for the feature-only model. It has not
been trained; inventing its numbers would be fabrication.

---

## 1. Final verdict on feature-only enhancement

**Verdict: WEAK feasibility for faithful enhancement. It is not a sound main
architecture for this project.**

It is *not* impossible to make it output a plausible-looking picture. It is
impossible for it to output **the enhanced version of that specific image**
reliably, because the input does not contain the information required to
reconstruct the scene:

1. **The mapping is massively under-determined [THEORY].** 14 scalars must
   determine 3×224×224 = **150 528 pixel values** (or ~608 400 at the project's
   600-px width). The 14 features are *global order-invariant summaries*: they
   count and average, they do not say **where** anything is.
2. **Images that are near-identical in feature space are completely different
   pictures [MEASURED — this repo, today].** Using the 14 currently selected
   features (standardised) on the 133 committed images, the closest pair
   (`UIEB_523.png` vs `UIEB_541.png`) sits at feature distance **0.420** where
   the *median* pair distance is **4.090** — i.e. ~10× closer than typical, and
   effectively indistinguishable to a model that sees only this vector. Those two
   images differ by **SSIM 0.1159** and **59.5 grey levels** mean absolute
   difference. A feature-only model must produce (essentially) the *same* output
   for both; the correct outputs differ enormously. It is therefore wrong for at
   least one of them, by roughly that margin — no training required to know this.
3. **Training converges to the conditional mean, and capacity cannot fix it
   [THEORY].** Minimising L2 loss over all functions *of the features* yields
   E[reference | features]; with L1, the conditional median. That is an *average
   picture*, so the output is blurry and structurally generic **by construction**.
   A larger network does not reduce this error — it changes nothing, because the
   missing information is missing from the input.
4. **The features are even weaker than "14 numbers" suggests [MEASURED — this
   repo, earlier].** 11 of the 25 were removed as redundant, including algebraic
   duplicates (`ASM = energy²`, `variance = std²`), so the effective
   dimensionality of the selected set is below 14.

Where the idea *does* have partial merit: a feature-only model can plausibly
learn **global photometric correction** — white balance, gamma, contrast, colour
cast — because those are exactly what global statistics measure. That is a
*tone-mapping* capability, not restoration. UIEB references are human-preferred
pseudo-references, so "typical enhanced look" is partly learnable; the specific
scene is not.

---

## 2. Information-loss analysis

| what is retained | what is lost |
|---|---|
| global colour statistics (mean R/G/B, `red_ratio`, `colorfulness`, saturation, value) → colour cast / red attenuation | **every spatial arrangement**: which pixel is where, object layout, where the edges are |
| global luminance statistics (`mean`, `std`, `variance`, `dynamic_range`, `rms_contrast`, `entropy`) → brightness/contrast/complexity | **Fourier phase**, which encodes structure; the features are essentially phase-blind |
| aggregate texture energy (GLCM family) → how textured, not where | texture *phase* and orientation detail (and this project's GLCM uses a single angle, `angles=[0]`) |
| edge/sharpness *counts* (`edge_density`, `gradient`, `laplacian_variance`, `keypoint_density`) | edge **positions**, contours, object identity, fine detail |
| a rough "quality fingerprint" of the current image | anything scene-specific: fish, coral, diver, texture layout, lighting direction |

**Two supporting measurements made today [MEASURED — this repo, today]:**

1. **Mirror test on 3 committed images** (`UIEB_1`, `UIEB_145`, `UIEB_97`): of the 25
   features, **23 are exactly invariant** under `hflip`/`vflip`/both (max relative
   deviation ≤ 2.6e-16). Two are not exactly invariant: `keypoint_density` moves
   up to **1.1 %** relative (5.9e-5 absolute) and `edge_density` up to **0.1 %**
   (4.9e-5 absolute). Of the 14 selected features, 13 are mirror-exact and only
   `keypoint_density` moves ~1 %.
   *Consequence:* the selected vector essentially cannot distinguish an image
   from its mirror — so a feature-only model must output (nearly) the same image
   for both, while the correct outputs are mirror images of each other.
   *Documentation note:* the repo states "all 25 features invariant to 2.2e-16
   (measured)"; today's measurement does not reproduce that for those two
   features (cv2 5.0.0 / skimage 0.26.0, the documented environment). It is a
   small effect, but the exact wording (and the augmentation-invariance note in
   `cnn/dataset.py`) should be corrected to "23 of 25 exactly; 2 within ~1 %".
2. **Feature-space collision test** (§1 item 2): different images can be ~10×
   closer than typical in the selected-feature space and still be unrelated
   pictures.

**Order-of-magnitude [THEORY, not measured]:** 14 float32 values = 56 bytes.
Even generously assuming ~8–16 *effective* bits per feature (they are smooth
statistics, several mutually redundant), the input carries on the order of
10²–10³ bits against an image of ~10⁵–10⁶ bytes. The bottleneck is not the
network; it is the representation.

---

## 3. Expected effect on PSNR and SSIM

**[EXPECTED — mechanism, no numbers invented]**

* **Structurally wrong output ⇒ SSIM low.** SSIM is explicitly sensitive to local
  structure and contrast. A conditional-mean/median image has no matching local
  structure, so SSIM against *that image's* reference should be poor.
* **PSNR limited by the same ambiguity.** Per-pixel error is dominated by the fact
  that the output is an "average scene" while the target is a specific scene.
* **The critical comparison is not against zero — it is against doing nothing.**
  On the sealed 133-image test split this project already has measured baselines
  **[MEASURED — this repo, earlier]**: raw (resize only) **SSIM 0.7603 / PSNR
  17.063 dB**; the classical pipeline **0.7636 / 17.089 dB**; the full-image U-Net
  **0.8003 / 19.324 dB**. A feature-only enhancer would have to beat 0.7603 SSIM
  just to be better than *not enhancing at all*. Since the correct output for a
  given input is a specific image it cannot see, that is very unlikely — it is
  more plausible that it scores *below* the trivial baseline. (This is an
  expectation, not a result; §12's experiment measures it.)
* **Blur is the expected failure mode**, and blur hurts both metrics — SSIM
  through lost structure, PSNR through smeared edges and wrong colours.

---

## 4. Expected effect on UIQM and UCIQE

**[EXPECTED — mechanism]** UIQM and UCIQE are *no-reference* metrics: they score
colourfulness, contrast, saturation and sharpness-like statistics of an image on
its own, with no reference. This creates a specific trap for a feature-only model:

* Because the model's **inputs are statistics**, and the NR metrics are
  **functions of statistics**, a network trained to reproduce the reference
  could produce images whose *statistics* match the "enhanced look" while the
  *scene* is wrong. Such outputs can score **well on UIQM/UCIQE but badly on
  PSNR/SSIM** — a plausible, even likely, divergence.
* The reverse can also happen (over-saturated/blurred images score poorly), which
  is why the **direction must be measured, never assumed**.
* **Reporting rule for the whole project:** UIQM/UCIQE are supporting evidence,
  never the headline. An enhanced image that "scores well" on a no-reference
  metric while not resembling the reference is a failed restoration. Any report
  where NR metrics look good and FR metrics look bad must say so explicitly and
  treat the NR result as suspect — and never compare our absolute UIQM/UCIQE with
  values printed in other papers (implementations differ).

---

## 5. Review / viva risk

Exactly the questions an examiner is likely to ask about **feature-only**
enhancement, in increasing order of danger:

1. "Your model never sees the image. Where does the image detail come from?"
2. "You reduce the image to 14 numbers and ask for 150 528 pixels back. How is that determined?"
3. "Your own features cannot tell an image from its mirror (measured). How does the model know which way round the scene is?"
4. "Two different photos can have nearly identical feature vectors (you measured this). What does the model output for them?"
5. "Is the network restoring the scene, or hallucinating a plausible one?"
6. "If the output is structurally wrong but colourful, what do UIQM/UCIQE actually reward?"
7. "What happens if the model outputs the *average* enhanced image? What SSIM does that get versus simply returning the input?"
8. "Why not give the image to the CNN? What does the image lack that your features add?" *(the answer is: nothing — features are a lossy function of the image; see §10)*
9. "Feature selection is central to your story — but is it a selection for a task where features are the *wrong input*?"
10. "Is this defensible as an engineering choice, or was it chosen because it looks 'feature-driven'?"

These are answerable only with experimental evidence, and the honest answer today
is: the architecture's input is insufficient for the task it is given.

---

## 6. Recommended architecture

**Recommendation: keep the hybrid architecture you already locked — image branch
+ statistically selected feature branch → fusion → conditioned enhancement
decoder.** It is your option (A)/(B) family, it uses **one** model, it keeps
feature selection central, and it removes the information bottleneck by letting
the decoder see the pixels.

Why this one and not the alternatives:

| option | verdict |
|---|---|
| **Feature-only (this proposal)** | rejected: insufficient input (this report) |
| **(A) Image CNN + features → fusion → decoder** | **recommended** — this is the locked design; fusion at the bottleneck (288 → 256) conditions the decoder; skips carry spatial detail; features stay central |
| (B) image feature maps + features → attention/gating | a *variant of the same single model*: conditioning by FiLM/gating instead of addition. It is a reasonable **second-stage refinement only if** the A/B ablation shows the feature branch is being ignored. Otherwise it adds complexity with no evidence — and your faculty asked for minimal machinery |
| (C) image → CNN → learned attention over features | makes the *handcrafted* features optional and dilutes the statistical stage's role; the statistical contribution would no longer drive the model |
| (D) features → quality prediction, separate enhancement model | this is the project's *previous* framing; it does not answer an enhancement objective and would mean two models |

Honesty clause (important, must survive into the thesis): **the features are
expected to contribute modestly to enhancement.** A pure image CNN can do the
whole job — the existing full-image U-Net does exactly that (**SSIM 0.8003 /
19.324 dB**, [MEASURED — this repo, earlier]). Adding features to pixels has
already been measured in *this* project, in the quality-prediction setting:
hybrid vs image-only changed avg R² by **+0.0108**, inside the ±~0.15 bootstrap CI
**[MEASURED — this repo, earlier]**. That is the right prior for enhancement too:
a modest, worth-measuring effect — not a guaranteed boost. The feature branch's
justification is (i) the statistical study is the project's contribution,
(ii) explicit global statistics condition the decoder interpretably, and
(iii) any gain is *measured* by the A/B ablation, not asserted.

---

## 7. Architecture diagram (recommended model, one model)

```
                    UIEB raw image (890 pairs)                UIEB reference (target only)
                              │                                          │
                              ▼                                          │
        PREPROCESSING (frozen: resize 600 → gray-world → CLAHE(L)        │
                       → bilateral → adaptive gamma)                     │
                              │                                          │
              ┌───────────────┴───────────────┐                          │
              ▼                               ▼                          │
     25 handcrafted features          letterbox 224×224×3                │
              │                               │                          │
   STATISTICAL SELECTION (train only)         │                          │
   F-test → BH-FDR → effect size → redundancy │                          │
              ▼                               │                          │
       14 SELECTED FEATURES                   │                          │
              │                               │                          │
              ▼                               ▼                          │
      ┌───────────────┐             ┌────────────────────┐               │
      │ FEATURE BRANCH│             │   IMAGE BRANCH      │              │
      │ StandardScaler│             │ Conv 3→32  112×112  │              │
      │ (train only)  │             │ Conv 32→64  56×56   │── skips s1,s2,s3
      │ Linear 14→32  │             │ Conv 64→128 28×28   │               │
      │ ReLU, Dropout │             │ Conv 128→256 14×14  │               │
      │    32-D       │             │  ┌──────────┬─────┐ │               │
      └───────┬───────┘             │  ▼          ▼     │ │               │
              │                     │ GAP→256-D  spatial │ │               │
              │                     │           256×14×14 │              │
              └────────────┬────────┴──────┬──────────────┘               │
                           ▼               │                              │
                 CONCATENATE 256+32 = 288-D                              │
                           ▼                                              │
                 FUSION  Linear(288→256) → ReLU                          │
                           ▼                                              │
              broadcast 256 → add to the 256×14×14 bottleneck              │
                           ▼                                              │
        DECODER: 14→28 (+128×28×28) → 56 (+64×56×56) → 112 (+32×112×112)  │
                 → 224 → Conv1×1 → 3 channels → Sigmoid                 │
                           ▼                                              │
                 ENHANCED IMAGE 3×224×224 → un-letterbox → (600×H)        │
                           ▼                                              │
        ┌──────────────┬────────────┬─────────────┬──────────────┐        │
        ▼              ▼            ▼             ▼              │        │
      PSNR           SSIM         UIQM          UCIQE            └────────┘
   (vs reference) (vs reference) (no-ref)     (no-ref)       training loss (L1)
```

Rejected sketch, for contrast (why it fails): `14 features → MLP → decoder →
image`. Everything above the decoder is missing, including all spatial
information.

---

## 8. Tensor / data flow (branch by branch)

| step | input | output |
|---|---|---|
| preprocessing | raw BGR H×W | 600×H′ uint8 (heights 266–901) |
| features | 600×H′ image | 25 scalars (14 selected after Stage A) |
| CNN branch input | letterboxed | **3 × 224 × 224** |
| block 1 / 2 / 3 / 4 | 3×224×224 → … | 32×112×112 → 64×56×56 → 128×28×28 → **256×14×14** |
| GAP | 256×14×14 | **256** vector |
| spatial path | 256×14×14 | kept as the decoder bottleneck |
| feature branch input | 14 selected, StandardScaler **fit on train only** | **14** |
| feature branch | Linear(14→32) → ReLU → Dropout(0.2) | **32** |
| concatenate | 256 ⊕ 32 | **288** |
| fusion | Linear(288→256) → ReLU | **256** |
| conditioning | 256 broadcast to 14×14, added to bottleneck | 256×14×14 |
| decoder up 1/2/3/4 | 14→28 (+128 ch) → 56 (+64 ch) → 112 (+32 ch) → 224 | 128×28×28 → 64×56×56 → 32×112×112 → 16×224×224 |
| output head | Conv1×1 → 3, Sigmoid | **3 × 224 × 224** |
| geometry restore | crop letterbox bars, resize to (600, H) | 3 × 600 × H |
| metrics | vs aligned UIEB reference | PSNR, SSIM, UIQM, UCIQE |

**The feature-only variant would have exactly one input tensor, `[B, 14]`, and
would be asked to produce `[B, 3, 224, 224]`** — that single line is the whole
problem.

---

## 9. Why features are used even though the image is also used (viva-ready)

*"The image branch learns spatial detail — where things are. The feature branch
supplies explicit, interpretable, statistically vetted global properties of the
image: how strong the red-channel attenuation is, how much contrast and entropy
the scene has, how textured and how edge-rich it is. Those are exactly the
per-image quantities that decide how much colour and contrast correction is
appropriate, and they are the quantities our statistical stage proved to be
informative. The CNN could approximate some of them, but not all, and not
transparently — and with only 623 training pairs, telling the model these global
properties explicitly is a useful prior. The features condition the decoder; they
do not replace the pixels. And because they are explicit, we can say WHY the
model behaved differently on two images. Whether they measurably improve the
result is answered by our ablation: the same decoder trained with and without the
feature branch, evaluated on the sealed test split."*

---

## 10. Why an image model can never be worse than a feature-only model (one line for the review)

The features are a **deterministic function of the image**, so "all functions of
the features" is a **subset** of "all functions of the image". A CNN given the
image can, in principle, compute everything the feature-only model could — plus
the spatial information it lacks. Therefore **image input weakly dominates
feature-only input: feature-only can never have a strictly higher attainable
score**, and it forfeits all structural information. The only arguments for
features-only are interpretability or severe data scarcity, and both are better
served by the hybrid (features *plus* pixels).

---

## 11. Implementation feasibility (recommended architecture)

| aspect | assessment |
|---|---|
| data | 890 paired images → 623 train / 134 val / 133 test, already frozen; paired geometric augmentation (8 transforms) already implemented |
| code reuse | encoder, feature branch, letterbox, train-only scaler, paired dataset, loss/eval utilities already exist in the repo and are verified |
| parameters | counted by `count_params()` at build time (no estimate quoted here) — the existing 3-level U-Net is 472 259 |
| compute | **[MEASURED — this repo, earlier]** the *smaller* existing U-Net trained at 128² crops, 3 levels, on 2 CPU cores at **99.5 s/epoch**. The locked hybrid is 224² with a 4-stage decoder, so several times heavier per epoch. A full 30–80-epoch run is therefore many hours on this CPU; a GPU (Colab/Kaggle) reduces it to minutes |
| risk | no risky machinery: conv/BN/ReLU/pool/upsample/concat/sigmoid only |
| feature-only variant | even cheaper to train — but it would consume compute to demonstrate a foreseeable information failure; the same conclusion is reachable in minutes by the experiment in §12 |

---

## 12. Final recommendation, and the smallest useful experiment

**Recommendation: implement the locked hybrid (image + 14 features → fusion →
conditioned decoder). Do not replace it with feature-only enhancement.** That is
a change to the locked flow, so it is your decision — but the evidence above says
the feature-only variant cannot produce a faithful enhanced image, and the
examiner questions in §5 have no good answers.

**Do not build a second model to demonstrate this.** Two cheap experiments
answer it, and both respect "one proposed model":

**E1 — already run today [MEASURED — this repo, today, free].** (a) the mirror
test (23/25 features exactly invariant; `keypoint_density` ~1 %) and (b) the
feature-space collision test (different images at ~10× closer-than-median feature
distance still differ by SSIM 0.116). Together these show the feature vector does
not identify the scene. These can go straight onto a slide titled *"Why the model
must also see the image."*

**E2 — the feature-only ceiling (recommended next, ~15 min, no training).**
Protocol, with the interpretation rule fixed **before** running it:

1. freeze the 14 features as selected by the new statistical stage (train-only);
2. standardise them with train statistics;
3. for each of the 133 sealed test images, retrieve its *k* nearest training
   neighbours in that 14-dimensional space (*k* = 1 and *k* = 5);
4. use those neighbours' **reference images** as the outputs: (a) the 1-NN
   reference, (b) the pixel-mean of the 5 references;
5. score both against the test image's own reference with the frozen
   `src/iqa.py` definition → mean PSNR/SSIM + bootstrap CI.

*Why this bounds the feature-only idea:* any function of the features can, at
best, reproduce the conditional distribution of references given the features;
*k*-NN with train data approximates its central tendency without any training.
Whichever score comes out is therefore an estimate of the **ceiling** for
feature-only enhancement.
*Pre-registered interpretation:* if the E2 ceiling is well below the full-image
U-Net's measured **SSIM 0.8003 / PSNR 19.324 dB** on the same 133 images
[MEASURED — this repo, earlier] — and below the do-nothing baselines of 0.7603 /
17.063 dB — then feature-only enhancement cannot be the project's main model, and
E2 becomes a *justified design decision* in the thesis rather than an assumption.

**If your faculty explicitly asks for the feature-only comparison**, run E2 and
present it as the *information ceiling* experiment — never as a second proposed
model.

**What must not change without your approval:** the locked flow, the one-model
rule, the role of the 14 features, and enhancement remaining enhancement. This
report proposes no change to any of them; it argues *against* one.

**WAITING for your decision.** No implementation code will be generated until you
approve the architecture: (A) keep the locked hybrid — recommended; or (B) run
E2 first and decide on its numbers.
