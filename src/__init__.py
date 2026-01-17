"""
UIT-DSC 2025 Challenge B - Hallucination Detection in Vietnamese
Source code package for training and inference
"""

__version__ = "1.0.0"
__author__ = "UIT-DSC 2025 Team"

from . import config
from . import utils
from . import tokenizer
from . import retriever
from . import model
from . import augmentation
from . import training
from . import inference
from . import metrics

__all__ = [
    "config",
    "utils", 
    "tokenizer",
    "retriever",
    "model",
    "augmentation",
    "training",
    "inference",
    "metrics",
]
