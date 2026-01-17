"""
Metrics and evaluation for UIT-DSC Challenge B
Computes various performance metrics and calibration scores
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple, Optional
from . import config

class MetricsComputer:
    """Compute various evaluation metrics"""
    
    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 class_names: Optional[list] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray,
               y_scores: Optional[np.ndarray] = None) -> Dict:
        """
        Compute all metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores/probabilities
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro')
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        metrics['per_class'] = {}
        for c in range(self.num_classes):
            y_true_binary = (y_true == c).astype(int)
            y_pred_binary = (y_pred == c).astype(int)
            
            metrics['per_class'][self.class_names[c]] = {
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'support': (y_true == c).sum()
            }
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Calibration metrics (if scores provided)
        if y_scores is not None:
            metrics['calibration'] = self._compute_calibration(y_true, y_scores)
        
        return metrics
    
    def _compute_calibration(self, y_true: np.ndarray, 
                           y_scores: np.ndarray) -> Dict:
        """Compute calibration metrics"""
        y_pred = y_scores.argmax(axis=-1)
        
        # Expected Calibration Error (ECE)
        probs = np.max(y_scores, axis=-1)
        correct = (y_pred == y_true).astype(float)
        
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
            if mask.sum() > 0:
                bin_acc = correct[mask].mean()
                bin_conf = probs[mask].mean()
                ece += np.abs(bin_acc - bin_conf) * mask.sum()
        
        ece /= len(y_true)
        
        return {'ece': float(ece)}
    
    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Print classification report"""
        print(classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        ))
    
    def confusion_matrix_normalized(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> np.ndarray:
        """Get row-normalized confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        return cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

class TemperatureScaling:
    """Temperature scaling for calibration"""
    
    def __init__(self, grid_start: float = 0.85,
                 grid_end: float = 1.35,
                 grid_step: float = 0.02):
        self.grid = np.arange(grid_start, grid_end + 1e-9, grid_step)
        self.temperatures = None
    
    def nll(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """Negative log-likelihood"""
        logp = logits - logits.max(axis=1, keepdims=True)
        logp = logp - np.log(np.exp(logp).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(y_true)), y_true].mean()
    
    def fit(self, logits: np.ndarray, y_true: np.ndarray,
           num_iters: int = 2) -> np.ndarray:
        """
        Fit per-class temperatures
        
        Args:
            logits: Model logits
            y_true: True labels
            num_iters: Number of iterations
        
        Returns:
            Temperature vector
        """
        num_classes = logits.shape[1]
        T = np.ones(num_classes, dtype=np.float32)
        
        for _ in range(num_iters):
            for c in range(num_classes):
                best_t, best_nll = T[c], float('inf')
                
                for t in self.grid:
                    T_try = T.copy()
                    T_try[c] = t
                    adj = logits / T_try[None, :]
                    nll = self.nll(adj, y_true)
                    
                    if nll < best_nll:
                        best_nll, best_t = nll, t
                
                T[c] = best_t
        
        self.temperatures = T
        return T
    
    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits"""
        if self.temperatures is None:
            return logits
        return logits / self.temperatures[None, :]

def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute PR-AUC (Precision-Recall AUC)"""
    from sklearn.metrics import auc, precision_recall_curve
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

if __name__ == "__main__":
    print("Metrics module loaded")
