"""
Text Cleaning Utilities for Reddit Sentiment Analysis
Cleans Reddit posts/comments while preserving financial terminology.
"""

import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RedditTextCleaner:
    """
    Cleans Reddit text for sentiment analysis.
    Preserves financial terms (important for FinBERT).
    """
    
    # URL pattern
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    # Emoji pattern (basic)
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    
    # Excessive punctuation (3+ consecutive)
    EXCESSIVE_PUNCT = re.compile(r'[!?.]{3,}')
    
    def __init__(self, min_words: int = 10):
        """
        Initialize text cleaner.
        
        Args:
            min_words: Minimum word count to keep text
        """
        self.min_words = min_words
    
    def clean(self, text: str) -> Optional[str]:
        """
        Clean text for sentiment analysis.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text, or None if text is too short/invalid
        """
        if not text or not isinstance(text, str):
            return None
        
        # Step 1: Remove URLs
        text = self.URL_PATTERN.sub(' ', text)
        
        # Step 2: Remove emojis
        text = self.EMOJI_PATTERN.sub(' ', text)
        
        # Step 3: Remove excessive punctuation (keep single punctuation)
        text = self.EXCESSIVE_PUNCT.sub(' ', text)
        
        # Step 4: Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Step 5: Lowercase (FinBERT can handle case, but lowercase is more consistent)
        text = text.lower()
        
        # Step 6: Check minimum word count
        word_count = len(text.split())
        if word_count < self.min_words:
            return None
        
        # Step 7: Check if text is mostly tickers/spam (heuristic)
        # If text is very short after cleaning or mostly symbols, skip
        if len(text) < 20:
            return None
        
        return text
    
    def construct_text(self, record: Dict) -> Optional[str]:
        """
        Construct and clean text from a Reddit record.
        
        Args:
            record: Record dict with title, selftext (submissions) or body (comments)
            
        Returns:
            Cleaned text, or None if invalid
        """
        if record.get("type") == "submission":
            raw_text = f"{record.get('title', '')} {record.get('selftext', '')}"
        elif record.get("type") == "comment":
            raw_text = record.get("body", "")
        else:
            # Fallback
            raw_text = (
                record.get("title", "") + " " +
                record.get("selftext", "") + " " +
                record.get("body", "")
            )
        
        return self.clean(raw_text)

