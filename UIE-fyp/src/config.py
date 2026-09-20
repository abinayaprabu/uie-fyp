"""Central configuration — single source of truth for paths, seeds and parameters.

Every script imports from here so that the whole pipeline (preprocessing,
feature extraction, ranking, CNN) uses identical settings. Paths are relative
to the project root (the folder containing this ``src`` directory).
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw-890"
PREPROCESSED_DIR = DATASET_DIR / "preprocessed"
REFERENCE_DIR = DATASET_DIR / "reference-890"

RESULTS_DIR = ROOT / "results"
FEATURE_RESULTS_DIR = RESULTS_DIR / "feature"
CNN_RESULTS_DIR = RESULTS_DIR / "cnn"
COMPARISON_RESULTS_DIR = RESULTS_DIR / "comparison"
PLOTS_DIR = ROOT / "plots"
MODELS_DIR = ROOT / "models"

FEATURE_DATASET_CSV = ROOT / "feature_dataset.csv"
SPLIT_CSV = RESULTS_DIR / "feature" / "data_split.csv"
VALIDATION_REPORT_CSV = RESULTS_DIR / "feature" / "dataset_validation_report.csv"

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# Data split (by image identity; the SAME ids are used by every stage).
# 70% train / 15% validation / 15% test  ->  train 623 / val 134 / test 133.
# (Verified against results/feature/data_split.csv. An earlier comment listed
#  these as "623 / 133 / 134", transposing val and test.)
# ----------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# ----------------------------------------------------------------------------
# Preprocessing (classical pipeline — see src/preprocess.py for definitions).
# ----------------------------------------------------------------------------
RESIZE_WIDTH = 600          # aspect-preserving resize; height follows
RESIZE_INTERP = "area"      # cv2.INTER_AREA (best for downsampling)
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID = (8, 8)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
GAMMA_MIN = 0.5             # adaptive gamma is clipped to [0.5, 2.0]
GAMMA_MAX = 2.0

# ----------------------------------------------------------------------------
# Feature ranking / selection (leakage-free: fitted on TRAIN split only).
# ----------------------------------------------------------------------------
RF_N_ESTIMATORS = 200
RF_MIN_SAMPLES_LEAF = 2
PERM_N_REPEATS = 5
CORR_THRESHOLD = 0.90       # abs(Pearson r) >= 0.90  ->  redundant
SUBSET_SIZES = (15, 12, 10, 8, 6)  # evaluated top-k subsets (rank order)

# ----------------------------------------------------------------------------
# CNN (see cnn/model.py). Input size is chosen AFTER inspecting the
# preprocessed dimensions; letterbox = aspect-preserving resize + padding.
# ----------------------------------------------------------------------------
CNN_INPUT_SIZE = 224        # square input (H = W = 224), letterboxed
CNN_BATCH_SIZE = 16
CNN_MAX_EPOCHS = 80
CNN_PATIENCE = 12           # early-stopping patience (epochs, on val loss)
CNN_LR = 1e-3
CNN_WEIGHT_DECAY = 1e-4
CNN_SEED = 42
# Extra seeds for repeated runs (H5: a single seed on a 133-image test split
# cannot separate model differences from run-to-run noise). The first entry is
# always CNN_SEED so single-seed runs stay comparable with earlier results.
CNN_SEEDS = (42, 43, 44)
CNN_N_BOOTSTRAP = 4000      # resamples for the test-R^2 percentile CI

# ---------------------------------------------------------------------------
# Augmentation (train split only). RESTRICTED to the Klein four-group
# {identity, hflip, vflip, hflip+vflip} because these four transforms were
# MEASURED to leave all 25 handcrafted features exactly invariant (max relative
# deviation 2.2e-16) and leave SSIM/PSNR invariant, since the same rigid
# transform applied to both images of a pair changes neither metric. So the
# 4x expansion needs no relabelling and no feature recomputation.
#
# 90-degree rotations are EXCLUDED: they convert the GLCM's horizontal
# adjacency (angles=[0]) into vertical adjacency, shifting the 8 GLCM features
# by up to 3.8%. They only become safe if GLCM is averaged over 4 angles.
# Photometric augmentation (brightness/contrast/colour/gamma jitter) is
# EXCLUDED outright: it alters the preprocessed image without altering the
# reference, so the SSIM/PSNR targets genuinely change and the labels become
# wrong.
# ---------------------------------------------------------------------------
CNN_AUGMENT_FLIPS = True

# The 25 handcrafted features in canonical order.
FEATURE_NAMES_25 = [
    # statistical
    "mean", "std", "variance", "entropy", "dynamic_range", "rms_contrast",
    # colour
    "mean_red", "mean_green", "mean_blue", "colorfulness", "red_ratio",
    "mean_saturation", "mean_value",
    # texture / GLCM
    "contrast", "correlation", "energy", "homogeneity", "ASM",
    "dissimilarity", "glcm_entropy", "glcm_variance",
    # edge / sharpness
    "edge_density", "gradient", "laplacian_variance", "keypoint_density",
]

TARGETS = ("ssim", "psnr")

# ===========================================================================
# ENHANCEMENT / RESTORATION U-NET  (image -> image)
# ---------------------------------------------------------------------------
# This is the component the project objective requires and that the quality
# -prediction CNNs (above) do NOT provide: those regress an image to two
# scalars [SSIM, PSNR], whereas this network maps a degraded image to an
# enhanced IMAGE of the same dimensions.
#
# The values below are "Config A", FROZEN on 2026-09-16 from a measured
# one-epoch benchmark (docs/enhancement-benchmark.md) and fixed BEFORE any
# training run. They are not to be tuned against the test split:
#   crop 128x128, batch 8, base channels 32 -> 472,259 trainable parameters,
#   61.9 s/epoch on 2 CPU cores, 1,290 MB peak RSS, 78 iterations/epoch.
# Nothing above this line is modified by the enhancement work.
# ===========================================================================
ENH_RUN_TAG = "unet_128"
ENH_CROP_SIZE = 128            # random crop, train only (val/test are full-res)
ENH_BATCH_SIZE = 8
ENH_BASE_CHANNELS = 32         # 3-level U-Net: 32 -> 64 -> 128
ENH_EXPECTED_PARAMS = 472_259  # asserted at build time; guards the frozen arch
ENH_LR = 2e-4                  # Adam; standard for L1 image restoration
ENH_WEIGHT_DECAY = 0.0
ENH_MAX_EPOCHS = 100           # FIXED BEFORE TRAINING - not adjusted afterwards
ENH_VAL_EVERY = 5              # full-resolution validation pass every N epochs
ENH_PATIENCE = 4               # in VALIDATIONS (= 20 epochs), on val SSIM
ENH_SEED = 42
# Paired geometric augmentation (train split only): flips x rot90 = 8 transforms
# applied IDENTICALLY to the input and the reference target, so the pair stays
# consistent. Unlike the quality-prediction pipeline, 90-degree rotations ARE
# safe here because no handcrafted/GLCM features are involved. Photometric
# augmentation is EXCLUDED: jittering the input without the target would teach
# the network a mapping that is wrong by construction.
ENH_AUGMENT_GEOMETRIC = True
ENH_LOSS = "l1"                # pixel-space reconstruction loss

ENHANCED_DIR = DATASET_DIR / "enhanced-test"       # gitignored (derived images)
ENHANCEMENT_RESULTS_DIR = RESULTS_DIR / "enhancement"
