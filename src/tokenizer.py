"""
Tokenization and text preprocessing for UIT-DSC Challenge B
Handles Vietnamese word segmentation and token preparation
"""

import re
import math
from typing import List, Optional, Tuple, Dict
import numpy as np
from collections import Counter

from . import config

# Try to import PyVi for Vietnamese word segmentation
try:
    from pyvi.ViTokenizer import tokenize as vi_tokenize
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False
    print("[WARN] PyVi not available - using raw text tokenization")

# ============== Basic Tokenization ==============
WORD_PATTERN = re.compile(r'\w+', re.UNICODE)
SENTENCE_PATTERN = re.compile(r'(?<=[\.\!\?\;])\s+|\n+')

def simple_tokenize(text: str) -> List[str]:
    """Simple word tokenization using regex"""
    if text is None:
        return []
    text = str(text).lower()
    return WORD_PATTERN.findall(text)

def segment_text(text: str) -> str:
    """
    Vietnamese word segmentation using PyVi
    Falls back to raw text if PyVi not available
    """
    if text is None or text == "":
        return ""
    
    if PYVI_AVAILABLE and config.USE_WORD_SEG:
        return vi_tokenize(text)
    else:
        return text

def split_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    if text is None or text == "":
        return []
    sentences = SENTENCE_PATTERN.split(text)
    return [s.strip() for s in sentences if s.strip()]

# ============== IDF & Retriever ==============
class IDFBuilder:
    """Build IDF (Inverse Document Frequency) from corpus"""
    
    def __init__(self, tokenize_fn=simple_tokenize):
        self.tokenize_fn = tokenize_fn
        self.doc_freq = Counter()
        self.num_docs = 0
        self.idf = {}
    
    def build(self, texts: List[str], min_df: int = 1):
        """
        Build IDF from list of texts
        
        Args:
            texts: List of documents
            min_df: Minimum document frequency threshold
        """
        # Count document frequency
        for text in texts:
            tokens = set(self.tokenize_fn(text))
            self.doc_freq.update(tokens)
        
        self.num_docs = len(texts)
        
        # Compute IDF
        self.idf = {}
        for word, freq in self.doc_freq.items():
            if freq >= min_df:
                self.idf[word] = math.log((self.num_docs + 1) / (1 + freq)) + 1.0
    
    def get_idf(self, word: str, default: float = 1.0) -> float:
        """Get IDF value for a word"""
        return self.idf.get(word, default)
    
    def get_idf_dict(self) -> Dict[str, float]:
        """Get full IDF dictionary"""
        return self.idf.copy()

# ============== Sentence Scoring for Retrieval ==============
def score_sentence_by_idf(sentence: str, query_tokens: set, 
                         idf_dict: Dict[str, float], view: int = 0) -> float:
    """
    Score a sentence based on query tokens and IDF
    
    Args:
        sentence: Sentence to score
        query_tokens: Set of query tokens
        idf_dict: IDF dictionary
        view: 0=idf, 1=idf+numeric, 2=idf+caps
    
    Returns:
        Score value
    """
    tokens = simple_tokenize(sentence)
    score = 0.0
    
    # Check capitalization (for view=2)
    has_cap = (sentence and sentence[0].isupper()) or \
              any(w and w[0].isupper() for w in sentence.split() if w)
    
    for token in tokens:
        if token in query_tokens:
            idf = idf_dict.get(token, 1.0)
            
            # Boost for numbers (view=1)
            if view == 1 and any(c.isdigit() for c in token):
                idf *= 1.5
            
            # Boost for capitalization (view=2)
            if view == 2 and has_cap:
                idf *= 1.2
            
            score += idf
    
    return score

def select_context_by_idf(context: str, prompt: str, response: str,
                         max_tokens: int, idf_dict: Dict[str, float],
                         neighbor: int = 1, view: int = 0) -> str:
    """
    Select relevant sentences from context using IDF scoring
    
    Args:
        context: Full context text
        prompt: User prompt
        response: LLM response
        max_tokens: Maximum tokens to keep
        idf_dict: IDF dictionary
        neighbor: Include neighbor sentences
        view: Retriever view (0, 1, or 2)
    
    Returns:
        Selected context text
    """
    sentences = split_sentences(context)
    if not sentences:
        return context
    
    # Query tokens from prompt and response
    query_tokens = set(simple_tokenize(prompt + " " + response))
    
    # Score all sentences
    scored = [(i, score_sentence_by_idf(s, query_tokens, idf_dict, view))
              for i, s in enumerate(sentences)]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Select sentences with neighbor support
    picked = set()
    total_tokens = 0
    
    for idx, _ in scored:
        # Add neighbor sentences
        neighbors = range(max(0, idx - neighbor), min(len(sentences), idx + neighbor + 1))
        
        for j in neighbors:
            if j not in picked:
                picked.add(j)
                # Estimate tokens ~ 1.2 * word_count
                total_tokens += len(simple_tokenize(sentences[j])) * 1.2
        
        if total_tokens >= max_tokens:
            break
    
    # Maintain sentence order
    picked_indices = sorted(picked)
    return " ".join(sentences[i] for i in picked_indices)

# ============== Length Estimation ==============
def estimate_p95_tokens(texts: List[str], percentile: float = 95.0) -> int:
    """
    Estimate p95 token length from texts
    
    Args:
        texts: List of text documents
        percentile: Percentile to compute (default 95)
    
    Returns:
        Estimated token count
    """
    lengths = [len(simple_tokenize(t)) for t in texts]
    return int(np.percentile(lengths, percentile))

# ============== Segment Masking ==============
class SegmentMasker:
    """Create segment masks for context/prompt/response"""
    
    def __init__(self, tokenizer, seg_tokens: Optional[List[str]] = None):
        self.tokenizer = tokenizer
        self.seg_tokens = seg_tokens or config.SEG_TOKENS
        
        # Add special tokens if not present
        if self.seg_tokens:
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": self.seg_tokens
            })
        
        self.ctx_token = self.tokenizer.convert_tokens_to_ids("[CTX]")
        self.prm_token = self.tokenizer.convert_tokens_to_ids("[PRM]")
        self.rsp_token = self.tokenizer.convert_tokens_to_ids("[RSP]")
    
    def create_masks(self, input_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create segment masks from input_ids
        
        Args:
            input_ids: Token IDs
        
        Returns:
            Tuple of (ctx_mask, prm_mask, rsp_mask)
        """
        B = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        ctx_mask = np.zeros((B, seq_len), dtype=np.int32)
        prm_mask = np.zeros((B, seq_len), dtype=np.int32)
        rsp_mask = np.zeros((B, seq_len), dtype=np.int32)
        
        for b in range(B):
            ids = input_ids[b]
            current_segment = None
            
            for i in range(seq_len):
                token_id = ids[i]
                
                if token_id == self.ctx_token:
                    current_segment = "ctx"
                elif token_id == self.prm_token:
                    current_segment = "prm"
                elif token_id == self.rsp_token:
                    current_segment = "rsp"
                elif current_segment == "ctx":
                    ctx_mask[b, i] = 1
                elif current_segment == "prm":
                    prm_mask[b, i] = 1
                elif current_segment == "rsp":
                    rsp_mask[b, i] = 1
        
        return ctx_mask, prm_mask, rsp_mask

# ============== Text Preprocessing ==============
class TextPreprocessor:
    """Unified text preprocessing pipeline"""
    
    def __init__(self, lowercase: bool = True, 
                 remove_extra_spaces: bool = True,
                 segment: bool = True):
        self.lowercase = lowercase
        self.remove_extra_spaces = remove_extra_spaces
        self.segment = segment
    
    def preprocess(self, text: str) -> str:
        """Preprocess text"""
        if text is None:
            return ""
        
        text = str(text)
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_extra_spaces:
            text = " ".join(text.split())
        
        if self.segment and PYVI_AVAILABLE:
            text = segment_text(text)
        
        return text

if __name__ == "__main__":
    # Test tokenization
    test_text = "Đây là một câu tiếng Việt để kiểm tra."
    print(f"Original: {test_text}")
    print(f"Tokens: {simple_tokenize(test_text)}")
    print(f"Segmented: {segment_text(test_text)}")
