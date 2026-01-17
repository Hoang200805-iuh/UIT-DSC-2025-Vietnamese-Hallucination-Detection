"""
Inference pipeline for UIT-DSC Challenge B
Implements TTA, fusion, and post-processing
"""

import torch
import numpy as np
from typing import Optional, List
from .utils import logger

@torch.no_grad()
def inference_single_batch(model, batch, device):
    """Single batch inference"""
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    outputs = model(**batch)
    logits = outputs.logits if isinstance(outputs, dict) else outputs.logits
    
    return logits.cpu().numpy()

@torch.no_grad()
def inference_with_tta(model, dataloader, num_passes: int = 9,
                      device: str = 'cuda'):
    """Test Time Augmentation (TTA) inference"""
    logits_list = []
    
    for _ in range(num_passes):
        all_logits = []
        model.eval()
        
        for batch in dataloader:
            batch_logits = inference_single_batch(model, batch, device)
            all_logits.append(batch_logits)
        
        logits_list.append(np.concatenate(all_logits, axis=0))
    
    # Average across TTA passes
    final_logits = np.mean(logits_list, axis=0)
    return final_logits

def fuse_logits(main_logits: np.ndarray,
               aux_logits: np.ndarray,
               main_weight: float = 0.8,
               aux_weight: float = 0.2) -> np.ndarray:
    """Fuse main and auxiliary logits"""
    return main_weight * main_logits + aux_weight * aux_logits

def apply_per_class_bias(logits: np.ndarray,
                        bias: np.ndarray) -> np.ndarray:
    """Apply per-class bias adjustment"""
    return logits + bias[np.newaxis, :]

def apply_temperature_scaling(logits: np.ndarray,
                             temperatures: np.ndarray) -> np.ndarray:
    """Apply per-class temperature scaling"""
    return logits / temperatures[np.newaxis, :]

def logits_to_predictions(logits: np.ndarray,
                         id2label: dict) -> List[str]:
    """Convert logits to class predictions"""
    pred_ids = logits.argmax(axis=-1)
    return [id2label[id] for id in pred_ids]

def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities using softmax"""
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

class InferencePostProcessor:
    """Post-process predictions with various techniques"""
    
    def __init__(self, id2label: dict,
                 use_temperature_scaling: bool = True,
                 use_per_class_bias: bool = True):
        self.id2label = id2label
        self.use_temperature_scaling = use_temperature_scaling
        self.use_per_class_bias = use_per_class_bias
        
        self.temperatures = None
        self.class_bias = None
    
    def set_temperatures(self, temperatures: np.ndarray):
        """Set temperature scaling values"""
        self.temperatures = temperatures
    
    def set_class_bias(self, bias: np.ndarray):
        """Set per-class bias values"""
        self.class_bias = bias
    
    def process(self, logits: np.ndarray) -> dict:
        """
        Post-process logits
        
        Args:
            logits: Model logits [N, C]
        
        Returns:
            Dictionary with predictions and scores
        """
        processed_logits = logits.copy()
        
        # Temperature scaling
        if self.use_temperature_scaling and self.temperatures is not None:
            processed_logits = apply_temperature_scaling(
                processed_logits, self.temperatures
            )
        
        # Per-class bias
        if self.use_per_class_bias and self.class_bias is not None:
            processed_logits = apply_per_class_bias(
                processed_logits, self.class_bias
            )
        
        # Get predictions and probabilities
        probs = logits_to_probabilities(processed_logits)
        preds = logits_to_predictions(processed_logits, self.id2label)
        pred_scores = probs.max(axis=-1)
        
        return {
            'predictions': preds,
            'probabilities': probs,
            'scores': pred_scores,
            'logits': processed_logits
        }

if __name__ == "__main__":
    print("Inference module loaded")
