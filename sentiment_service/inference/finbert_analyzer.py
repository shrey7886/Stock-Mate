"""
FinBERT Sentiment Analyzer
Uses ProsusAI/finbert model for financial sentiment analysis.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FinBERTAnalyzer:
    """
    Financial sentiment analyzer using FinBERT model.
    Outputs continuous sentiment scores: positive_prob - negative_prob
    """
    
    MODEL_NAME = "ProsusAI/finbert"
    LABELS = ["positive", "negative", "neutral"]
    
    def __init__(self, device: Optional[str] = None, batch_size: int = 32):
        """
        Initialize FinBERT analyzer.
        
        Args:
            device: Device to run model on ("cuda", "cpu", or None for auto)
            batch_size: Batch size for inference
        """
        self.batch_size = batch_size
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading FinBERT model on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with sentiment probabilities and composite score
        """
        if not text or len(text.strip()) < 10:
            return {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "sentiment_score": 0.0
            }
        
        # Truncate to model's max length (512 tokens)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Map probabilities to labels
        result = {
            "positive": float(probs[0]),
            "negative": float(probs[1]),
            "neutral": float(probs[2]),
            "sentiment_score": float(probs[0] - probs[1])  # continuous score
        }
        
        return result
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment for a batch of texts (more efficient).
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment dictionaries
        """
        if not texts:
            return []
        
        # Filter empty texts
        valid_texts = [(i, t) for i, t in enumerate(texts) if t and len(t.strip()) >= 10]
        
        if not valid_texts:
            return [{
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "sentiment_score": 0.0
            }] * len(texts)
        
        results = [None] * len(texts)
        
        # Process in batches
        for batch_start in range(0, len(valid_texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(valid_texts))
            batch_indices, batch_texts = zip(*valid_texts[batch_start:batch_end])
            
            # Tokenize batch
            inputs = self.tokenizer(
                list(batch_texts),
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # Store results
            for idx, prob in zip(batch_indices, probs):
                results[idx] = {
                    "positive": float(prob[0]),
                    "negative": float(prob[1]),
                    "neutral": float(prob[2]),
                    "sentiment_score": float(prob[0] - prob[1])
                }
        
        # Fill in None results (empty texts)
        for i in range(len(results)):
            if results[i] is None:
                results[i] = {
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 1.0,
                    "sentiment_score": 0.0
                }
        
        return results

