import pandas as pd
from typing import Dict, List
import re

class SocialMediaAnalyzer:
    """Analyze sentiment of social media posts"""
    
    def __init__(self):
        # Sentiment lexicon
        self.positive_words = {
            "bullish", "buy", "moon", "gains", "pump", "long", "hold",
            "diamond", "hands", "strong", "buy", "love", "best", "great",
            "awesome", "excellent", "good", "positive", "up", "higher"
        }
        self.negative_words = {
            "bearish", "sell", "crash", "dump", "short", "loss", "panic",
            "weak", "bad", "horrible", "terrible", "worst", "down", "lower",
            "negative", "avoid", "poor", "disappointed", "hate"
        }
        
        # Emoji sentiment indicators
        self.positive_emojis = ["🚀", "📈", "💰", "💎", "👍", "✅", "🎉"]
        self.negative_emojis = ["📉", "💔", "😞", "❌", "👎", "🔴", "⚠"]
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of social media text"""
        text_lower = text.lower()
        words = text_lower.split()
        
        # Count sentiment words
        pos_count = sum(1 for word in words if word.lower() in self.positive_words)
        neg_count = sum(1 for word in words if word.lower() in self.negative_words)
        
        # Count emojis
        pos_emoji_count = sum(1 for emoji in self.positive_emojis if emoji in text)
        neg_emoji_count = sum(1 for emoji in self.negative_emojis if emoji in text)
        
        total_sentiment_indicators = pos_count + neg_count + pos_emoji_count + neg_emoji_count
        
        if total_sentiment_indicators == 0:
            composite_score = 0.0
        else:
            composite_score = (pos_count + pos_emoji_count - neg_count - neg_emoji_count) / max(1, total_sentiment_indicators)
            # Clamp to [-1, 1]
            composite_score = max(-1.0, min(1.0, composite_score))
        
        return {
            "positive_count": pos_count + pos_emoji_count,
            "negative_count": neg_count + neg_emoji_count,
            "composite_score": composite_score
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple social media posts"""
        return [self.analyze_sentiment(text) for text in texts]