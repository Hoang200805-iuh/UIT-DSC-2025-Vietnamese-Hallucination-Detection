"""
Training pipeline for UIT-DSC Challenge B
Implements training loops and callbacks
"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainerCallback
from typing import Optional, Dict, Any
import numpy as np
from .utils import logger

class WeightedTrainer(Trainer):
    """Custom trainer with weighted loss and auxiliary outputs"""
    
    def __init__(self, use_ema: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_ema = use_ema
        self.ema = None
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute weighted loss"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits if isinstance(outputs, dict) else outputs.logits
        
        # Main loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        
        # Auxiliary losses (if model has _aux)
        if hasattr(model, '_aux') and model._aux:
            aux = model._aux
            
            # Hall loss
            if 'logit_hall' in aux:
                labels_binary = (labels > 0).float()
                hall_loss = nn.BCEWithLogitsLoss()(aux['logit_hall'], labels_binary)
                loss += 0.2 * hall_loss
            
            # IE loss
            if 'logit_ie' in aux:
                labels_ie = ((labels == 2).float())
                ie_loss = nn.BCEWithLogitsLoss()(aux['logit_ie'], labels_ie)
                loss += 0.3 * ie_loss
        
        return (loss, outputs) if return_outputs else loss

class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback"""
    
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_metric = None
        self.wait_count = 0
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check metric and stop if needed"""
        if metrics is None:
            return
        
        current_metric = metrics.get(f"{args.metric_key_prefix}_loss", None)
        
        if current_metric is None:
            return
        
        if self.best_metric is None or current_metric < self.best_metric:
            self.best_metric = current_metric
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                logger.info(f"Early stopping triggered. Best metric: {self.best_metric}")
                control.should_training_stop = True

class LoggingCallback(TrainerCallback):
    """Callback for detailed logging"""
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log at each step"""
        if state.global_step % args.logging_steps == 0:
            logger.info(f"Step {state.global_step}: Loss = {state.loss:.4f}")

def get_optimizer(model, learning_rate: float = 2e-5,
                 weight_decay: float = 0.01):
    """Get optimizer with proper weight decay"""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    from torch.optim import AdamW
    return AdamW(optimizer_grouped_parameters, lr=learning_rate)

def get_scheduler(optimizer, num_training_steps: int,
                 warmup_steps: int = None):
    """Get learning rate scheduler"""
    if warmup_steps is None:
        warmup_steps = int(num_training_steps * 0.1)
    
    from transformers import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

if __name__ == "__main__":
    print("Training module loaded")
