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
# 70% train / 15% validation / 15% test  ->  623 / 133 / 134 images.
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
