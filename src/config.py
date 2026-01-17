"""
Configuration module for UIT-DSC Challenge B
Centralize all hyperparameters and settings
"""

import os
from pathlib import Path

# ============== Project Structure ==============
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if not exist
for d in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, CONFIG_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)

# ============== Data Paths ==============
TRAIN_PATH = DATA_DIR / "vihallu-train.csv"
TEST_PATH = DATA_DIR / "vihallu-public-test.csv"
PRIVATE_TEST_PATH = DATA_DIR / "vihallu-private-test.csv"

# ============== Model Configuration ==============
MODEL_NAME = "vinai/phobert-base"
TOKENIZER_NAME = "vinai/phobert-base"
MAX_LEN = 256  # PhoBERT max position embeddings

# ============== Classes ==============
LABEL_TO_ID = {"no": 0, "intrinsic": 1, "extrinsic": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
NUM_CLASSES = len(LABEL_TO_ID)
CLASS_NAMES = list(LABEL_TO_ID.keys())

# ============== Training Configuration ==============
SEED = 42
FOLDS = 5
EPOCHS = 5
BASE_BATCH_SIZE = 6
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
LABEL_SMOOTHING = 0.02
GRADIENT_ACCUMULATION_STEPS = 2

# Learning rate scheduler
USE_COSINE_SCHEDULE = True
USE_WARMUP = True

# ============== Optimization & Augmentation ==============
USE_FP16 = True
USE_BF16 = False
ENABLE_GRAD_CHECKPOINT = True

# R-Drop
USE_RDROP = True
RDROP_ALPHA = 0.20

# FGM (Fast Gradient Method)
USE_FGM = True
FGM_EPS = 1.0
FGM_EMB_NAME = "embeddings.word_embeddings"

# EMA (Exponential Moving Average)
USE_EMA = True
EMA_DECAY = 0.9999

# ============== Retriever Configuration ==============
USE_RETRIEVER = True
RETRIEVER_VIEW = 0  # 0: idf, 1: idf+numeric, 2: idf+caps
USE_NEIGHBOR = True
NEIGHBOR_K = 1
PR_FLOOR_RATIO = 0.85

# ============== Custom Head Configuration ==============
USE_CUSTOM_HEAD = True
USE_CORR_FEATURES = True
CORR_MLP_DIMS = 18
CORR_MLP_HIDDEN = 64

# Two-stage head (Hall + IE)
USE_TWO_STAGE_HEAD = True
HALL_LAMBDA = 0.20
IE_LAMBDA = 0.30

# IE losses
USE_IE_FOCAL = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.6

USE_IE_SUPCON = True
SUPCON_LAMBDA = 0.10
SUPCON_TAU = 0.07

USE_IE_MARGIN = True
IE_MARGIN_M = 0.5
IE_MARGIN_LAMBDA = 0.20
USE_IE_MARGIN_SCHEDULE = True
IE_MARGIN_M_START = 0.45
IE_MARGIN_M_END = 0.80

# ============== Segment Tokens ==============
ADD_SEG_TOKENS = True
SEG_TOKENS = ["[CTX]", "[PRM]", "[RSP]"]

# Column names
CTX = "context"
PRM = "prompt"
RSP = "response"
ID_COL = "id"

# ============== Test Time Augmentation (TTA) ==============
USE_TTA = True
TTA_PASSES = 9

# ============== Output & Evaluation ==============
OUTPUT_FORMAT = {
    "id": "id",
    "predict_label": "predict_label"
}

# Temperature scaling
USE_TEMPERATURE_SCALING = True
TEMP_SCALING_ITERS = 2
TEMP_SCALING_GRID_START = 0.85
TEMP_SCALING_GRID_END = 1.35
TEMP_SCALING_GRID_STEP = 0.02

# Fusion weights (for multi-head inference)
FUSION_MAIN_WEIGHT = 0.8
FUSION_AUX_WEIGHT = 0.2

# Per-class bias adjustment
CLASS_BIAS = [-0.05, 0.02, 0.02]  # [no, intrinsic, extrinsic]

# ============== Class Balanced Loss ==============
USE_CB_WEIGHTS = True
CB_BETA = 0.9999

# ============== Logging & Checkpointing ==============
LOGGING_STEPS = 50
EVAL_STEPS = None  # Use epoch-based eval
SAVE_STRATEGY = "epoch"
LOAD_BEST_MODEL = True

# ============== Device ==============
DEVICE = "cuda"  # Use cuda if available

# ============== Number of Workers ==============
NUM_WORKERS = 2  # For DataLoader
NUM_PROC = 2     # For HF datasets preprocessing

# ============== Features & Preprocessing ==============
USE_WORD_SEG = True  # Vietnamese word segmentation using PyVi
USE_SEGMENT_MASKS = True  # Separate masks for ctx, prm, rsp

def get_config_dict():
    """Return configuration as dictionary"""
    return {
        # Model
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "num_classes": NUM_CLASSES,
        
        # Training
        "seed": SEED,
        "folds": FOLDS,
        "epochs": EPOCHS,
        "batch_size": BASE_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        
        # Features
        "use_retriever": USE_RETRIEVER,
        "use_custom_head": USE_CUSTOM_HEAD,
        "use_tta": USE_TTA,
        "tta_passes": TTA_PASSES,
    }

if __name__ == "__main__":
    print("Configuration Summary:")
    for key, value in get_config_dict().items():
        print(f"  {key}: {value}")
