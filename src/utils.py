"""
Utility functions for UIT-DSC Challenge B
Helper functions for data processing, logging, and common operations
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import logging

from . import config

# ============== Logging Setup ==============
def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if log_file:
        log_path = config.LOGS_DIR / log_file
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.StreamHandler()]
        )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ============== Seed Setting ==============
def set_all_seeds(seed: int):
    """
    Set seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"All seeds set to {seed}")

# ============== File I/O ==============
def save_json(data: Dict[str, Any], path: str):
    """Save dictionary to JSON file"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved JSON to {path}")

def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {path}")
    return data

def save_metrics(metrics: Dict[str, Any], filename: str = "metrics.json"):
    """Save metrics to output directory"""
    output_path = config.OUTPUT_DIR / filename
    save_json(metrics, str(output_path))

# ============== Device Management ==============
def get_device():
    """Get torch device (cuda if available, else cpu)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA - Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

# ============== Text Processing ==============
def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text"""
    if text is None:
        return ""
    return " ".join(text.split())

def clean_text(text: str) -> str:
    """Basic text cleaning"""
    if text is None:
        return ""
    text = normalize_whitespace(text)
    text = text.strip()
    return text

# ============== Feature Computation ==============
def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets"""
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union

def coverage(set_a: set, set_b: set) -> float:
    """Compute coverage: how much of set_a is in set_b"""
    if len(set_a) == 0:
        return 0.0
    return len(set_a & set_b) / len(set_a)

# ============== Model Utilities ==============
def count_parameters(model) -> int:
    """Count total trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model) -> float:
    """Get model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def log_model_info(model, model_name: str = "Model"):
    """Log model information"""
    num_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    logger.info(f"{model_name}:")
    logger.info(f"  Trainable Parameters: {num_params:,}")
    logger.info(f"  Model Size: {model_size:.2f} MB")

# ============== Checkpoint Management ==============
def save_checkpoint(model, optimizer, epoch: int, 
                   save_path: Optional[str] = None):
    """Save model checkpoint"""
    if save_path is None:
        save_path = config.OUTPUT_DIR / f"checkpoint_epoch_{epoch}.pt"
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path}")

def load_checkpoint(model, optimizer, checkpoint_path: str):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
    return epoch

# ============== Data Info ==============
def log_data_info(df, name: str = "Dataset"):
    """Log dataset information"""
    logger.info(f"{name}:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    if "label" in df.columns:
        logger.info(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

# ============== Prediction Utilities ==============
def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities using softmax"""
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

def apply_temperature_scaling(logits: np.ndarray, 
                             temperatures: np.ndarray) -> np.ndarray:
    """Apply per-class temperature scaling to logits"""
    return logits / temperatures[None, :]

class AverageMeter:
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    # Test utilities
    set_all_seeds(42)
    print("Utilities loaded successfully")
