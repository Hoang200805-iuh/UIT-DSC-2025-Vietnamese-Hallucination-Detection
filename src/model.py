"""
Model architecture for UIT-DSC Challenge B
Implements custom hallucination detection head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from . import config

class CorrelationMLP(nn.Module):
    """MLP for processing correlation features"""
    
    def __init__(self, input_dim: int = 18, hidden_dim: int = 64, 
                 output_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.mlp(x)

class HallucinationDetector(nn.Module):
    """
    Custom hallucination detection model with dual-head architecture
    
    Features:
    - Base transformer model
    - Custom head with correlation features
    - Two-stage classification (Hall + IE)
    - Inference fusion logic
    """
    
    def __init__(self, base_model, num_classes: int = 3,
                 use_corr_features: bool = True,
                 use_two_stage: bool = True,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.base = base_model
        hidden_size = base_model.config.hidden_size
        
        self.use_corr_features = use_corr_features
        self.use_two_stage = use_two_stage
        
        # Correlation MLP (18 dims -> 32 dims)
        if use_corr_features:
            self.corr_mlp = CorrelationMLP(
                input_dim=18,
                hidden_dim=64,
                output_dim=32
            )
            classifier_input_dim = hidden_size + 32
        else:
            classifier_input_dim = hidden_size
        
        # Dropout layers
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate) for _ in range(3)
        ])
        
        # Main classifier
        self.classifier_main = nn.Linear(classifier_input_dim, num_classes)
        
        # Two-stage head (Hall + IE)
        if use_two_stage:
            self.classifier_hall = nn.Linear(classifier_input_dim, 1)  # Hallucination detection
            self.classifier_ie = nn.Linear(classifier_input_dim, 1)    # Intrinsic/Extrinsic
            
            # IE representation for contrastive learning
            ie_repr_dim = hidden_size // 4
            self.ie_repr_proj = nn.Linear(2 * hidden_size, ie_repr_dim)
        
        self._aux = {}
    
    def mean_pool(self, last_hidden_state, attention_mask):
        """Mean pooling with attention mask"""
        attention_mask = attention_mask.unsqueeze(-1).float()
        masked_output = last_hidden_state * attention_mask
        sum_output = masked_output.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1)
        return sum_output / (sum_mask + 1e-10)
    
    def forward(self, input_ids, attention_mask, 
                global_corr: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            global_corr: Global correlation features [B, 5]
            labels: Target labels [B]
        
        Returns:
            SequenceClassifierOutput with logits
        """
        # Base model forward
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        last_hidden = outputs.last_hidden_state
        
        # Mean pooling
        pooled = self.mean_pool(last_hidden, attention_mask)
        
        # Prepare classifier input
        if self.use_corr_features:
            # Prepare correlation features
            # In actual implementation, these would be computed from seg_masks
            if global_corr is not None:
                corr_feats = torch.cat([
                    torch.zeros((pooled.size(0), 13), device=pooled.device),
                    global_corr
                ], dim=-1)
            else:
                corr_feats = torch.zeros((pooled.size(0), 18), device=pooled.device)
            
            corr_emb = self.corr_mlp(corr_feats)
            classifier_input = torch.cat([pooled, corr_emb], dim=-1)
        else:
            classifier_input = pooled
        
        # Main classification
        logits = torch.stack([
            self.classifier_main(d(classifier_input))
            for d in self.dropouts
        ], dim=0).mean(dim=0)
        
        # Two-stage classification (for training)
        if self.use_two_stage and self.training:
            logit_hall = self.classifier_hall(classifier_input).squeeze(-1)
            logit_ie = self.classifier_ie(classifier_input).squeeze(-1)
            self._aux = {
                "logit_hall": logit_hall,
                "logit_ie": logit_ie,
            }
        
        # Inference fusion (for evaluation)
        elif self.use_two_stage and not self.training:
            logit_hall = self.classifier_hall(classifier_input).squeeze(-1)
            logit_ie = self.classifier_ie(classifier_input).squeeze(-1)
            
            # Fusion: Hall + IE -> 3-class
            p_hall = torch.sigmoid(logit_hall)
            p_ie = torch.sigmoid(logit_ie)
            
            p_no = 1 - p_hall
            p_intr = p_hall * (1 - p_ie)
            p_extr = p_hall * p_ie
            
            p_aux = torch.stack([p_no, p_intr, p_extr], dim=-1).clamp(1e-6, 1-1e-6)
            logits_aux = torch.log(p_aux)
            
            # Fused logits with per-class bias
            logits = logits * 0.8 + logits_aux * 0.2
            bias = torch.tensor([-0.05, 0.02, 0.02], 
                               dtype=logits.dtype, device=logits.device)
            logits = logits + bias
        
        return {
            "logits": logits,
            "aux": self._aux if hasattr(self, '_aux') else {}
        }

def create_model(model_name: str = config.MODEL_NAME,
                num_classes: int = config.NUM_CLASSES,
                use_corr_features: bool = config.USE_CORR_FEATURES,
                use_two_stage: bool = config.USE_TWO_STAGE_HEAD):
    """
    Create hallucination detection model
    
    Args:
        model_name: Pretrained model name
        num_classes: Number of output classes
        use_corr_features: Whether to use correlation features
        use_two_stage: Whether to use two-stage head
    
    Returns:
        HallucinationDetector model
    """
    from transformers import AutoModel
    
    base_model = AutoModel.from_pretrained(model_name)
    model = HallucinationDetector(
        base_model,
        num_classes=num_classes,
        use_corr_features=use_corr_features,
        use_two_stage=use_two_stage
    )
    
    return model

if __name__ == "__main__":
    print("Model architecture module loaded")
