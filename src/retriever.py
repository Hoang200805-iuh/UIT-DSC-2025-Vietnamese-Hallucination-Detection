"""
Retriever module for UIT-DSC Challenge B
Implements advanced context retrieval with multiple views
"""

from typing import Optional, Dict, List
import numpy as np
from . import config
from .tokenizer import IDFBuilder, score_sentence_by_idf, select_context_by_idf

class Retriever:
    """
    Advanced retriever with multiple views and strategies
    
    Supports:
    - View 0: Basic IDF scoring
    - View 1: IDF with numeric boost
    - View 2: IDF with capitalization boost
    """
    
    def __init__(self, idf_dict: Optional[Dict[str, float]] = None):
        """
        Initialize retriever
        
        Args:
            idf_dict: Pre-computed IDF dictionary
        """
        self.idf_dict = idf_dict or {}
    
    def build_idf(self, contexts: List[str], min_df: int = 1):
        """Build IDF from contexts"""
        builder = IDFBuilder()
        builder.build(contexts, min_df=min_df)
        self.idf_dict = builder.get_idf_dict()
    
    def retrieve(self, context: str, prompt: str, response: str,
                max_tokens: int = 200, neighbor: int = 1,
                view: int = 0) -> str:
        """
        Retrieve relevant context sentences
        
        Args:
            context: Full context text
            prompt: User prompt
            response: LLM response
            max_tokens: Maximum tokens to keep
            neighbor: Number of neighboring sentences to include
            view: Retriever view (0, 1, or 2)
        
        Returns:
            Selected context
        """
        if not context or not self.idf_dict:
            return context
        
        return select_context_by_idf(
            context, prompt, response,
            max_tokens, self.idf_dict,
            neighbor=neighbor, view=view
        )

class DynamicRetriever:
    """
    Dynamic retriever that adjusts budget based on input
    """
    
    def __init__(self, idf_dict: Optional[Dict[str, float]] = None,
                 base_budget: int = 200):
        self.retriever = Retriever(idf_dict)
        self.base_budget = base_budget
    
    def compute_budget(self, prompt: str, response: str,
                      floor_ratio: float = 0.85) -> int:
        """
        Compute dynamic budget based on prompt+response length
        
        Args:
            prompt: User prompt
            response: LLM response
            floor_ratio: Minimum budget ratio
        
        Returns:
            Budget (max tokens for context)
        """
        pr_tokens = len(prompt.split()) + len(response.split())
        # Conservative: budget = base - pr_tokens, with floor
        budget = max(
            int(self.base_budget * floor_ratio),
            self.base_budget - pr_tokens
        )
        return budget
    
    def retrieve_dynamic(self, context: str, prompt: str, response: str,
                        view: int = 0) -> str:
        """
        Retrieve with dynamic budget
        
        Args:
            context: Full context text
            prompt: User prompt
            response: LLM response
            view: Retriever view
        
        Returns:
            Selected context
        """
        budget = self.compute_budget(prompt, response)
        return self.retriever.retrieve(
            context, prompt, response,
            max_tokens=budget, view=view
        )

if __name__ == "__main__":
    # Test retriever
    print("Retriever module loaded")
