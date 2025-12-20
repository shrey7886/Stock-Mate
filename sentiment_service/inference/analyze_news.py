import pandas as pd
from typing import Dict, List

class NewsAnalyzer:
    """Analyze sentiment of financial news using lexicon-based approach"""
    
    def __init__(self):
        """Initialize NewsAnalyzer with sentiment lexicon"""
        # Sentiment lexicon for fallback
        self.positive_words = {
            "gain", "profit", "surge", "soar", "rally", "bullish", "outperform",
            "strong", "growth", "success", "advance", "boom", "record", "surge",
            "beat", "exceed", "rose", "jump", "jumped", "positive", "upbeat",
            "upgrade", "upside"
        }
        self.negative_words = {
            "loss", "decline", "crash", "plunge", "bearish", "underperform",
            "weak", "fall", "fell", "down", "slump", "miss", "disappoint",
            "negative", "downbeat", "tumble", "drop", "fell", "slump",
            "downgrade", "downside"
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text"""
        return self._analyze_lexicon(text)
    
    def _analyze_lexicon(self, text: str) -> Dict[str, float]:
        """Use lexicon-based sentiment analysis"""
        text_lower = (text or "").lower()
        words = text_lower.split()
        
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        total = len(words)
        
        if total == 0:
            return {
                "negative": 0.0,
                "neutral": 1.0,
                "positive": 0.0,
                "composite_score": 0.0
            }
        
        pos_score = pos_count / total
        neg_score = neg_count / total
        neu_score = max(0.0, 1.0 - pos_score - neg_score)
        
        return {
            "negative": neg_score,
            "neutral": neu_score,
            "positive": pos_score,
            "composite_score": pos_score - neg_score
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts"""
        return [self.analyze_sentiment(text) for text in texts]