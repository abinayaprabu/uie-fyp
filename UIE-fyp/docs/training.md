# Training — the feature-guided enhancer, exactly as it is run

*Written 2026-10-02. Configuration lives in `src/config.py` (the single source
of truth, block "FEATURE-GUIDED ENHANCEMENT — Stage B/C"); the code lives in
`cnn/feature_guided/`. Nothing in this document is aspirational — each setting
below is what the drivers actually use.*

## 1. The two runs

| | image-only (ablation) | feature-guided (proposed) |
|---|---|---|
| command | `python -m cnn.feature_guided.train --variant image_only --threads 2` | `python -m cnn.feature_guided.train --variant feature_guided --threads 2` |
| run tag | `enh224_imgonly` | `enh224_featguided` |
| feature branch | off (ignores its input) | on, FiLM at the bottleneck |
| parameters | 1,170,963 | 1,188,211 |
| everything else | identical: data, split, augmentation, loss, LR, batch, seeds, epochs, early stopping | identical |

The two commands differ in exactly one flag. That is what makes the ablation
interpretable: any measured difference is attributable to the feature
conditioning branch (and to nothing else).

## 2. Data

- Pairs: 890 UIEB underwater/reference pairs; the frozen split is
  **train 623 / validation 134 / test 133** (`results/feature/data_split.csv`,
  seed 42, group-aware by SHA-256 of the raw bytes so duplicate images never
  straddle splits).
- Input: the **preprocessed** underwater image (600-px width pipeline),
  letterboxed to 224×224.
- Target: the aligned UIEB **reference**, letterboxed with the same geometry.
- Handcrafted features: the 10 selected features, read from
  `results/feature/feature_quality_dataset.csv` (full-resolution values,
  matching the image branch) and standardised with a scaler fitted on the
  **train split only** at run start.
- The test split is never loaded by the trainer; the drivers print an
  identifier fingerprint for train/val/test so a run can be tied to the exact
  id lists.

## 3. Model

Encoder: `Conv3×3 → BN → ReLU → MaxPool2` blocks with widths 32/64/128/256,
spatial sizes 224→112→56→28→14. Feature branch: `Linear(k→32) → ReLU →
Dropout(0.2) → Linear(32→2·256)` → (γ, β); FiLM `F·(1+γ)+β` on the 256×14×14
bottleneck. FiLM weights are initialised to zero, so at step 0 the guided model
is *exactly* the image-only model. Decoder: 256→128→64→32→16 with 2× bilinear
upsampling and skip connections (128×28×28, 64×56×56, 32×112×112), then a
1×1/3×3 convolution to 3 channels and a sigmoid (range [0, 1]).

The modulation is written ``F·(1+γ) + β`` (standard FiLM). The specification's
``γ·F + β`` is the same family under the reparameterisation γ' = 1 + γ; the
``1 +`` form is used because it makes the initialisation exactly the identity
(γ = β = 0), which is what makes the ablation start from a level field.

## 4. Optimisation

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Loss | L1 (primary). `L1 + λ·(1 − SSIM)` available via `HYBRID_LOSS=l1_ssim`; λ has no default and must be chosen on train/val — **never quoted from a paper** |
| Epochs | ≤ 80 |
| Validation | full-resolution PSNR/SSIM every 2 epochs |
| Model selection | best validation **SSIM** (a best-val-loss checkpoint is also kept) |
| Early stopping | patience 12 epochs without validation-SSIM improvement |
| Seed | 42 (python/numpy/torch) |
| Threads | 2 (this sandbox) |
| Checkpoints | `models/best_<tag>.pt` plus a byte-identical durable copy `results/enhancement/<tag>/best_<tag>.pt` (kept in git). Both are rewritten at EVERY validation improvement, so an interrupted multi-hour run still leaves the best model on disk |

Augmentation: paired horizontal/vertical/both flips only (the Klein four
group), applied identically to input and target and **never** to validation or
test. The features were measured to be invariant under these flips (23/25
exactly, 2 within 6e-5), so cached features stay valid. No photometric
augmentation: it would change the feature values and the colour statistics the
model is asked to correct.

## 5. Pre-flight evidence (required before full training)

1. `python scripts/test_hybrid_shapes.py` — encoder/decoder shapes, output
   range, FiLM identity at init (`max|guided − image_only| = 0`), non-zero
   conditioning gradients after two Adam steps, feature dependence under
   shuffling/zeroing (max |Δ| 1.5e-2 / 3.0e-2), parameter counts, timing.
   **All pass.**
2. Tiny overfit — training loss must fall sharply on a handful of images.
   Two runs are kept as evidence: 12 images / 6 epochs
   (`results/enhancement/smoke_overfit/`, 0.2404 -> 0.1597) and 6 images /
   30 epochs (`results/enhancement/smoke_overfit6_long/`, 0.2618 -> 0.1003,
   monotonically decreasing). Both use `--val-every 0 --no-save`.
3. `python scripts/leakage_audit.py` — 12 checks; the checkpoint-dependent
   checks re-run automatically once `models/best_enh224_*.pt` exist.
4. `python scripts/test_nr_metrics.py` — UIQM/UCIQE sanity.

## 6. Cost (measured)

~150 s per epoch on this 2-core sandbox for 39 iterations of batch 16
(validation epochs add the full-resolution pass). The 80-epoch cap is therefore
≈ 3.3 h per run; early stopping usually ends sooner. The two runs were launched
sequentially in the background; checkpoints are committed so a sandbox reset
cannot destroy them.

## 7. Reproducing a run

```bash
# 1. (already done, verified byte-identical) frozen data pipeline
python scripts/run_preprocessing.py
python scripts/build_feature_dataset.py
# 2. statistics + selected features
python scripts/run_statistics.py && python scripts/make_feature_selection_outputs.py
# 3. pre-flight
python scripts/test_hybrid_shapes.py
python -m cnn.feature_guided.train --variant feature_guided --run-tag smoke_overfit \
    --limit-train 6 --batch 4 --epochs 6 --val-every 0 --no-save
# 4. the two real runs
python -m cnn.feature_guided.train --variant image_only  --threads 2
python -m cnn.feature_guided.train --variant feature_guided --threads 2
# 5. enhance the sealed test split and evaluate
python -m cnn.feature_guided.enhance --run enh224_featguided --split test
python scripts/verify_final_model.py --run enh224_featguided   # checkpoint + features + forward pass
python scripts/evaluate_hybrid.py                              # sealed-test table + panels
```

(If the optional image-only ablation was trained too, enhance and evaluate it
with the same two commands and its row appears automatically.)

## 8. What the training log records

`results/enhancement/<tag>/train_history.csv` — per epoch: train loss, wall
time, validation SSIM/PSNR/L1, best flag; `run_config.json` records the exact
configuration, the git commit, the split fingerprints and the scaler
statistics, so no run can be silently re-interpreted later.
