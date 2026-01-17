"""
Augmentation and regularization techniques for UIT-DSC Challenge B
Implements R-Drop, FGM, EMA, SupCon, and margin-based losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class RDropLoss(nn.Module):
    """R-Drop: Regularized Dropout Loss (Liang et al., 2021)"""
    
    def __init__(self, alpha: float = 0.5, ce_loss_fn=None):
        super().__init__()
        self.alpha = alpha
        self.ce_loss_fn = ce_loss_fn or nn.CrossEntropyLoss()
    
    def forward(self, logits1, logits2, labels):
        """
        Compute R-Drop loss
        
        Args:
            logits1: First forward pass logits
            logits2: Second forward pass logits (with dropout)
            labels: Target labels
        
        Returns:
            Combined loss
        """
        ce_loss = self.ce_loss_fn(logits1, labels)
        
        # KL divergence between two distributions
        p1 = F.softmax(logits1, dim=-1)
        p2 = F.softmax(logits2, dim=-1)
        
        kl_loss = F.kl_div(F.log_softmax(logits2, dim=-1), p1, reduction='bsxmean')
        kl_loss += F.kl_div(F.log_softmax(logits1, dim=-1), p2, reduction='bsxmean')
        
        return ce_loss + self.alpha * kl_loss / 2

class FGM:
    """Fast Gradient Method for adversarial training"""
    
    def __init__(self, model, eps: float = 1.0, 
                 emb_name: str = "embeddings.word_embeddings"):
        self.model = model
        self.eps = eps
        self.emb_name = emb_name
        self.backup = {}
    
    def attack(self):
        """Perturb embeddings"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if param.grad is not None:
                    self.backup[name] = param.data.clone()
                    norm = torch.norm(param.grad)
                    if norm != 0:
                        r_at = self.eps * param.grad / norm
                        param.data.add_(r_at)
    
    def restore(self):
        """Restore original embeddings"""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

class EMA:
    """Exponential Moving Average for model weights"""
    
    def __init__(self, model, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.ema_model = None
        self.init_ema()
    
    def init_ema(self):
        """Initialize EMA model"""
        import copy
        self.ema_model = copy.deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    def update(self):
        """Update EMA weights"""
        with torch.no_grad():
            for param, ema_param in zip(
                self.model.parameters(),
                self.ema_model.parameters()
            ):
                ema_param.data = self.decay * ema_param.data + \
                                 (1 - self.decay) * param.data
    
    def apply_to(self, model):
        """Apply EMA weights to model"""
        if self.ema_model is None:
            return
        
        for param, ema_param in zip(
            model.parameters(),
            self.ema_model.parameters()
        ):
            param.data = ema_param.data

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss (Khosla et al., 2020)"""
    
    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau
    
    def forward(self, features, labels):
        """
        Compute SupCon loss
        
        Args:
            features: Feature vectors [B, D]
            labels: Target labels [B]
        
        Returns:
            Loss value
        """
        B = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        # Similarity matrix
        sim_matrix = torch.mm(features, features.t()) / self.tau
        
        # Create mask for positive pairs (same label)
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask.fill_diagonal_(0)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        neg = exp_sim.sum(dim=1, keepdim=True)
        pos = (mask * exp_sim).sum(dim=1, keepdim=True)
        
        loss = -torch.log(pos / (neg + 1e-6))
        return loss.mean()

class MarginLoss(nn.Module):
    """Margin-based loss with scheduling"""
    
    def __init__(self, margin: float = 0.5, lambda_weight: float = 0.1):
        super().__init__()
        self.margin = margin
        self.lambda_weight = lambda_weight
    
    def forward(self, logits, labels, current_margin: Optional[float] = None):
        """
        Compute margin loss
        
        Args:
            logits: Model logits [B, C]
            labels: Target labels [B]
            current_margin: Current margin value (for scheduling)
        
        Returns:
            Loss value
        """
        if current_margin is None:
            current_margin = self.margin
        
        # Get logits for target class
        target_logits = logits.gather(1, labels.unsqueeze(1))
        
        # Get max logits for other classes
        other_logits = logits.clone()
        other_logits.scatter_(1, labels.unsqueeze(1), -float('inf'))
        max_other_logits = other_logits.max(dim=1)[0]
        
        # Margin loss: max(0, margin + other_logit - target_logit)
        margin_loss = torch.clamp(
            current_margin + max_other_logits - target_logits,
            min=0
        ).mean()
        
        return margin_loss

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.6, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, labels):
        """
        Compute focal loss
        
        Args:
            logits: Model logits [B, C]
            labels: Target labels [B]
        
        Returns:
            Loss value
        """
        ce = F.cross_entropy(logits, labels, reduction='none')
        
        # Get probability of correct class
        probs = torch.exp(-ce)
        
        # Focal loss: -alpha * (1 - p)^gamma * ce
        focal = self.alpha * (1 - probs) ** self.gamma * ce
        
        return focal.mean()

def schedule_margin(current_epoch: int, total_epochs: int,
                   start_margin: float = 0.45,
                   end_margin: float = 0.80) -> float:
    """
    Linear margin scheduling
    
    Args:
        current_epoch: Current training epoch
        total_epochs: Total number of epochs
        start_margin: Starting margin value
        end_margin: Ending margin value
    
    Returns:
        Current margin value
    """
    progress = min(current_epoch / max(total_epochs, 1), 1.0)
    return start_margin + (end_margin - start_margin) * progress

if __name__ == "__main__":
    print("Augmentation module loaded")
