# Repo audit and build report (required first response)

*Written 2026-10-02 against branch `arena/01a0bd95-uie-fyp`, commit `d089c20`.
This is the §37 first-response report for the locked master-build prompt:
"Feature-Guided Underwater Image Enhancement Using Statistical Selection of
Handcrafted Image-Quality Features". Every number here was read from the
repository, not estimated.*

**Status in one line:** Stage A (statistical selection) is finished and
verified; Stage B code is written and passes its pre-flight tests; the
image-only baseline is training right now; nothing has been deleted; no claim
of improvement is made anywhere, because the ablation has not run yet.

---

## 1. The 14 required pre-implementation answers

### 1.1 Repository structure

```
UIE-fyp/
  src/       config.py, preprocess.py, features.py, iqa.py, metrics.py, nr_metrics.py
  stats/     descriptive, hypothesis_tests, multiple_testing, correlation,
             redundancy, selector            <- Stage A (new)
  cnn/       dataset.py, dataset_pairs.py, model.py, train.py, evaluate.py,
             predict.py, unet.py, train_enhance.py, enhance.py,
             train_hybrid.py, enhance_hybrid.py,
             hybrid/{encoder,conditioning,decoder,model,losses,dataset}   <- new
  scripts/   20 files (preprocessing, split, dataset build, statistics,
             leakage audit, shape/metric tests, baselines, evaluation, charts)
  docs/      8 pre-existing documents + this report
  results/   cnn/, comparison/, feature/, statistics/, enhancement/
  dataset/   raw-890/ (890), reference-890/ (890), enhanced-test/ (133),
             _uieb_mirror.tar.gz, mirror_file_listing.json
```

There is **no `configs/` directory**. The single source of truth for every
constant is `src/config.py`; §27 asks for `configs/` but creating a second
place for constants would contradict "one source of truth", so the existing
`src/config.py` is used and this deviation is stated rather than hidden.

### 1.2 Preprocessing files

`src/preprocess.py` — `resize_to_width` → `gray_world_white_balance` →
`apply_clahe(clip=2.5, grid 8×8)` → `apply_bilateral(9, 75, 75)` →
`adaptive_gamma(0.5…2.0)`; `enhance()` chains them; `resize_reference_like_preprocessed()`
reproduces the reference size convention. Driver `scripts/run_preprocessing.py`.
Frozen since Stage 1. **Re-run today on the freshly extracted 890 pairs and the
output `preprocessing_log.csv` came out byte-identical to the committed file.**

### 1.3 Feature files

`src/features.py::extract_features` computes the 25 handcrafted features (GLCM
at levels=256, single documented angle). `scripts/build_feature_dataset.py`
extracts them on the **full-resolution preprocessed image** and computes the
two reference metrics, then runs checks 3–10 and exits 1 on any failure.
Outputs: `results/feature/feature_quality_dataset.csv` (890 × 28 = image_name +
25 features + `ssim` + `psnr`) and `results/feature/preprocessing_log.csv`.
Also re-run today: **byte-identical** to the committed files.

### 1.4 Split / dataset files

`scripts/make_split.py` → `results/feature/data_split.csv` (890 rows):
group-aware 70/15/15 by SHA-256 of the raw bytes → **train 623 / val 134 /
test 133**, 7 duplicate groups kept whole. `cnn/dataset_pairs.py` aligns each
raw name with its reference and exposes `aligned_reference`, `split_ids`,
`ids_fingerprint`. `dataset/enhanced-test/` holds the 133 U-Net outputs of the
sealed test split (committed so they cannot be lost again).

### 1.5 Enhancement model

- Existing baseline (prior work): `cnn/unet.py::EnhancementUNet` (472,259
  params) trained by `cnn/train_enhance.py` on 128-px crops.
- Proposed model (this build): `cnn/feature_guided/` — encoder (32→64→128→256 at
  112/56/28/14), FiLM conditioning in the bottleneck, U-Net decoder with skips
  (14→28→56→112→224), 3-channel sigmoid output. **1,188,211 params** with the
  feature branch, **1,170,963** image-only (the ablation twin).
- The old `cnn/model.py` (288→128→2) is a *quality predictor*, not an
  enhancer; it stays prior work and is **not** reused as the proposed model.

### 1.6 Metrics

`src/iqa.py` — `compute_ssim_psnr(pre_bgr, ref_bgr)` (skimage, `data_range=255`,
reference first) plus independent `psnr_from_scratch` / `ssim_from_scratch`
cross-checks. `src/metrics.py` — regression metrics, bootstrap CIs, paired
Wilcoxon. `src/nr_metrics.py` — UIQM (Panetta) and UCIQE (Yang & Sowmya) with
every convention written in the module docstring; verified by
`scripts/test_nr_metrics.py`. `scripts/verify_enhancement.py` reproduces the
committed U-Net numbers bit-identically.

### 1.7 Training scripts

Prior work: `cnn/train.py` (quality-prediction models), `cnn/train_enhance.py`
(U-Net baseline). New: `cnn/feature_guided/train.py` (both variants, identical settings
except the branch) and `cnn/feature_guided/enhance.py` (inference for the sealed test
split). Both drivers refuse to run with mismatched split fingerprints.

### 1.8 Existing results

| Artefact | Content |
|---|---|
| `results/feature/` | labels, split, correlation matrix, OLD-14 list, ranking CSVs, validation report |
| `results/cnn/` | four prior *quality-prediction* models (RF / MLP / image-only / hybrid-all25 / hybrid-14) |
| `results/comparison/` | `baseline_metrics.csv`, `ablation_results.csv`, RF test predictions |
| `results/enhancement/` | `unet_128*` history + `metrics.json` + `best_*.pt`; `smoke_overfit/`; `enh224_imgonly/` (running) |
| `results/statistics/` | 11 Stage-A files (new) |

Measured on the sealed 133-image test split: raw SSIM 0.760276 / PSNR 17.063267;
classical 0.763648 / 17.088898; U-Net 0.800296 / 19.324221 (ΔSSIM +0.036648,
CI [0.028532, 0.045499], 102W/31L; ΔPSNR +2.2353 dB, CI [1.8781, 2.5879],
121W/12L).

No-reference metrics for the same three systems were computed for the first
time today by `scripts/evaluate_hybrid.py`
(`results/metrics/final_results.csv`, n = 133, same images):

| Method | PSNR | SSIM | UIQM | UCIQE |
|---|---|---|---|---|
| Raw | 17.0633 | 0.7603 | 7.3322 | 20.4920 |
| Classical | 17.0889 | 0.7636 | 7.7645 | 22.6355 |
| Existing U-Net | 19.3242 | 0.8003 | 7.6281 | 25.6052 |

Note the honest oddity: the U-Net is best on PSNR, SSIM **and** UCIQE but has a
lower UIQM than the classical pipeline. UIQM is not a distance-to-truth
measure, and this is exactly why the project reports both families and never
treats a higher no-reference score as proof of better enhancement.

Classical vs raw on this subset: ΔSSIM +0.0034 (CI [-0.0166, +0.0228],
76W/57L, p = 0.4338), ΔPSNR +0.0256 dB (CI [-0.7964, +0.8074]) — i.e. the
existing classical pipeline is **not** significantly better than the untouched
input on the sealed test split. That is worth knowing before anyone writes
"classical enhancement improves the image" in the report.

### 1.9 Files to archive or delete

**Nothing will be deleted without approval.** Proposal only: the old
quality-prediction track (`cnn/{model,train,predict,evaluate}.py`,
`scripts/{run_baselines,run_ablation,feature_ranking,feature_subset_evaluation,feature_correlation,make_plots}.py`,
`results/cnn/*`) is superseded for the final deliverable but explains months of
work and is the "prior work" the report compares against. Recommendation: keep
it in place, and add a `docs/legacy.md` index that says which files are prior
work; move only if you explicitly approve an archive move.

### 1.10 Exact modifications required (and made)

| Change | Files | Kind |
|---|---|---|
| Append-only Stage B/C constants | `src/config.py` (41 new lines) | done |
| Un-ignore durable checkpoints | `.gitignore` | done |
| Stage-A code | `stats/` (6 modules + `__init__`) | done |
| Stage-A driver + evidence | `scripts/run_statistics.py`, `results/statistics/` | done |
| Leakage audit | `scripts/leakage_audit.py` | done |
| Proposed model + drivers | `cnn/feature_guided/`, `cnn/feature_guided/train.py`, `cnn/feature_guided/enhance.py` | done |
| Pre-flight tests | `scripts/test_hybrid_shapes.py`, `scripts/test_nr_metrics.py` | done |
| Test-set evaluation + ablation + figures | `scripts/evaluate_hybrid.py`, `scripts/run_hybrid_ablation.py` | TODO |
| Documentation set (§29) | `docs/{architecture,statistical_analysis,feature_selection,training,metrics}_explanation.md`, `docs/viva_questions.md` | TODO |

No change to preprocessing, features, the split, or any committed label file.

### 1.11 Conflicts with the final architecture

1. `cnn/model.py` is a predictor whose output head must not be confused with
   the enhancer; names "hybrid_final" under `results/cnn/` refer to the OLD
   prediction experiment, not to this build.
2. The old U-Net baseline was trained on 128-px crops at full resolution; the
   new twins use 224-px letterboxed input. The comparison table must disclose
   this protocol difference (T1 handicap).
3. Two different "14-feature" meanings exist: the OLD committed list (prior
   selection) and the Stage-A candidate set. The OLD file is left untouched at
   `results/feature/final_selected_features.csv`.
4. "256" is overloaded in the old docs: encoder channels, the 256-d GAP
   vector, and GLCM levels. The new docs will spell out which is meant.
5. `docs/enhancement-benchmark.md` referenced from an old commit message does
   not exist; the U-Net timing (61.9 vs 99.5 s/epoch) is still unreconciled.
6. `results/comparison/baseline_metrics.csv` has empty CI cells for the neural
   models; the CIs live in `results/cnn/*/metrics.json` (`r2_ci95`).
7. Letterbox bars (T2, stated rather than hidden): the plan was to mask the
   L1 on the ~32 % of the 224×224 input that is padding. The implemented loss
   does **not** mask. The bars are black in both the letterboxed input and the
   letterboxed target, so the L1 there is a trivial, fast-saturating term, and
   every reported metric is computed after un-letterboxing at full resolution,
   so padding never enters PSNR/SSIM. Masking can still be added, but doing it
   now would invalidate the two in-flight twin runs, and it would have to be
   applied to both to keep the ablation fair. It is recorded here as a known
   deviation from the T2 plan, not as a solved item.

### 1.12 Proposed tensor shapes

| Stage | Shape | Note |
|---|---|---|
| raw → preprocessed | 600 × H (H 266…901) | frozen pipeline |
| model input | 3 × 224 × 224 | INTER_AREA letterbox + centred black pad |
| encoder skips | 32×112², 64×56², 128×28² | **pooled** maps |
| bottleneck | 256×14×14 | kept spatial |
| GAP vector | 256 | for the report only (not used to modulate) |
| features | 25 → K selected (K = 10 today) | from the FULL 600-px image |
| feature MLP | K → 32 (ReLU, dropout 0.2) | |
| conditioning | 32 → 256 γ and 256 β | FiLM, identity at init |
| conditioned bottleneck | 256×14×14 | `F·(1+γ)+β` |
| decoder | 14→28(+128)→56(+64)→112(+32)→224 | |
| output | 3×224×224, sigmoid | un-letterbox + resize to reference size |
| metrics | PSNR/SSIM (full ref) + UIQM/UCIQE (NR) | same protocol for every method |

### 1.13 Exact statistical pipeline (Stage A, already run)

All steps on train (623) for fitting/selection; validation (134) only for the
random-forest importance fit-and-score step; the sealed test was never read
(asserted in code and audited).

1. **Descriptive stats** on train and all rows (`descriptives_*.csv`) — context
   only, never selection.
2. **Per-feature f_regression vs SSIM and vs PSNR** (`hypothesis_tests.csv`):
   F, raw p, BH-adjusted q. Pearson r is reported as the effect size of the
   *same* test, not as independent evidence.
3. **BH-FDR** at α = 0.05 (`multiple_testing.py`); raw and adjusted p-values
   are both stored.
4. **Random-forest permutation importance** (fit on train, scored on val;
   `rf_permutation_importance.csv`) — separate SSIM and PSNR rankings.
5. **Combined rank** = mean of the two 1-based ranks; ties broken by mean
   normalised importance, then alphabetically (`selector.combine_rankings`).
6. **Redundancy removal** on TRAIN Pearson |r| ≥ 0.90, keeping the stronger
   ranked feature (`redundancy.py`, `redundant_pairs_train.csv`).
7. **Selected set** → `results/statistics/selected_features.csv` (+ narrative in
   `selection_summary.md`). Today this yields **10 features**, not the old 14 —
   see section 4 below; that decision is yours.

### 1.14 Expected computational cost (measured, not guessed)

| Job | Cost |
|---|---|
| Stage A statistics | seconds (CPU) |
| Preprocessing + features (890 images) | 60 s + 172 s |
| Proposed model, 1 epoch | ~150 s at batch 16 / 39 iterations (image-only, measured) |
| Full run, 80 epochs + early stop | ≈ 3.3 h per variant; the twins together ≈ 7 h |
| U-Net baseline (already done) | 99.5 s/epoch, ~124 min total |
| Full-res UIQM/UCIQE | 0.07–0.17 s per image, vectorised and verified |

**T4 decision.** The sandbox has 2 CPU cores. Uploading 1.6 GB of UIEB images
to Colab just to train two small models is not sensible, so the plan is: run the
twins locally in the background with durable checkpoints committed to git, and
keep the code Colab-ready (same command, `--threads` flag) in case a longer
block is ever available. The image-only run is doing exactly this now.

---

## 2. A–H audit

**A. Keep (unchanged, verified).** `src/preprocess.py`, `src/features.py`,
`src/iqa.py`, `src/metrics.py`, `scripts/build_feature_dataset.py`,
`scripts/make_split.py`, `cnn/dataset_pairs.py`, `cnn/unet.py` +
`cnn/train_enhance.py` (baseline), `dataset/enhanced-test/` (133 PNGs), the
label files, and the OLD-14 list.

**B. Modify (append-only).** `src/config.py`, `.gitignore`. Nothing existing
was edited, renumbered, or overwritten.

**C. Delete.** Nothing. See 1.9 for the archive *proposal* (needs approval).

**D. Create.** `stats/`, `scripts/run_statistics.py`, `scripts/leakage_audit.py`,
`cnn/feature_guided/`, `cnn/feature_guided/train.py`, `cnn/feature_guided/enhance.py`,
`src/nr_metrics.py`, `scripts/test_hybrid_shapes.py`,
`scripts/test_nr_metrics.py`, `results/statistics/`, this report. Still to
create: the hybrid evaluation/ablation drivers and the documentation set.

**E. Leakage risks found and handled.**

- Selection statistics / scaler / correlations: TRAIN ONLY (verified by
  `scripts/leakage_audit.py`, 12 checks, exit 0).
- Augmentation: applied to train only; val/test force `augment=False`
  (verified in the audit).
- References: the paired dataset class never reads a reference for training
  targets; the 60 challenging images are NR-only and were never paired.
- Split: identifiers are fingerprinted in the training log
  (`02317209a3a5acff` / `12a871f98f64b156` / `cae970a0ebbfb1ac`).
- Remaining checks (scaler == train-only; checkpoint fingerprint) activate
  automatically once `models/best_enh224_*.pt` exist — they must be re-run
  after training.

**F. Architectural inconsistencies found.** The seven items in 1.11.

**G. Reusable results.** The frozen split, the label file, the Stage-A tables,
the U-Net metrics/PNG evidence, the prior-quality-model numbers, and the 25
descriptive statistics are all reusable exactly as committed.

**H. Claims NOT to be reused.** "Features can replace the image" (measured and
rejected in `docs/feasibility-feature-only-enhancement.md`); the R²-based
reconstruction of ground-truth scores; "the hybrid beats image-only" (never
tested — the ablation is still pending); UIQM/UCIQE values compared against
other papers' tables; "our model is SOTA"; and the OLD 14-feature set being
"the" final selection.

---

## 3. Acceptance-criteria status (§0–§37)

| § | Requirement | Status |
|---|---|---|
| 0 | Audit before coding | DONE — `build-inventory.md` + this report |
| 1 | Semantics kept separate (features ≠ CNN activations) | DONE in code and docs |
| 2 | Fixed architecture diagram | DONE (docs + 1.12) |
| 3 | 25 features not silently replaced | DONE — extracted full 25, flagged variance/std, RMS-contrast/std, ASM/energy |
| 4 | 890 pairs; challenging-60 NR-only | DONE — no pair ever formed |
| 5 | Frozen split, no leakage, scaler train-only | DONE + audited |
| 6 | Frozen preprocessing | DONE — reproducible byte-identically today |
| 7 | 224 ≠ 600 ≠ 256 | DONE — documented |
| 8 | Encoder shapes, keep spatial maps | DONE — verified by the shape test |
| 9 | Features from the full preprocessed image | DONE — verified |
| 10 | f_regression primary, BH-FDR, Spearman robustness | DONE — `results/statistics/` |
| 11 | RF permutation on validation, separate rankings | DONE |
| 12 | Train-only |r| ≥ 0.90 redundancy | DONE — 19 pairs, 7 removals |
| 13 | Do not force 14; report OLD vs NEW | **OPEN — needs your decision (section 4)** |
| 14 | Small MLP → one explainable modulation | DONE — FiLM, identity at init |
| 15 | U-Net decoder with skips | DONE |
| 16 | L1 primary, optional λ·(1−SSIM) | DONE (λ = 0 in the main runs; configurable) |
| 17 | Train 623, early stop on 134, test once | RUNNING (image-only) |
| 18 | Paired-only augmentation | DONE — audited |
| 19 | PSNR/SSIM + UIQM/UCIQE from published definitions | DONE in code (`src/nr_metrics.py`) |
| 20 | Verify baselines before reuse; identical protocol | DONE — U-Net reproduced bit-identically |
| 21 | Required ablation, no pre-claiming | PENDING training |
| 22 | Qualitative analysis (best/median/failure) | TODO |
| 23 | Resolution handling documented | DONE |
| 24 | Pre-flight checklist before full training | DONE — shape test + 12-image overfit (loss 0.2404→0.1597) |
| 25 | Guidance sanity check (normal/shuffled/zeroed) | DONE in the shape test (max |Δ| 2.96e-2) |
| 26 | Leakage audit script | DONE — 12 checks pass |
| 27 | Modular layout, one source of truth | DONE (see the configs/ note in 1.1) |
| 28 | Archive obsolete experiments after preserving results | PROPOSAL only (1.9) |
| 29 | Documentation A–P + diagram | TODO — indexes exist, explanations pending |
| 30 | Distinct purpose per statistical method | DONE — no ANOVA/t-tests for complexity |
| 31 | Experiments E1–E10 | E1–E7 done or running; E8–E10 pending |
| 32 | Final table Method×PSNR/SSIM/UIQM/UCIQE | 3 of 5 rows measured (`results/metrics/final_results.csv`); hybrid rows produced automatically when the twins finish |
| 33 | No unfounded claims, no fabricated metrics | HONOURED throughout |
| 34 | Review material + viva doc | TODO |
| 35 | Strongest defensible model, report honestly | HONOURED |
| 36 | Execution order 1–22 | Steps 1–5 complete, step 6 in progress |
| 37 | This report | DELIVERED |

---

## 4. Open decisions (yours)

1. **Feature set (§13) — the one blocking question.**
   The statistical funnel that you specified produces **10 features**, not 14.
   OLD-14 (kept at `results/feature/final_selected_features.csv`) and NEW-10
   (`results/statistics/selected_features.csv`) are listed side by side in the
   chat message that accompanies this report. The model code takes the set
   size from the CSV, so either choice runs; adopting NEW-10 means the locked
   "14" in the prompt becomes a candidate set, exactly as §13 allows — but
   only with your explicit approval.
2. **T4 / compute (1.14).** Confirm the plan to run the 80-epoch twins locally
   (≈7 h total) rather than moving to Colab.
3. **Cleanup (1.9).** Confirm "keep everything, no archive move" or approve an
   archive move.
4. **Loss weighting.** The main runs use L1 only (λ = 0), as the prompt's
   primary setting. A single λ-tuned run can be added later if there is time;
   it will not be claimed to come from any paper.
