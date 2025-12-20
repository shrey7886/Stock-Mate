"""
Stock Mention Detection
Matches Reddit posts/comments to stock tickers using ticker symbols and company names.
"""

import re
from typing import Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)


class StockMatcher:
    """
    Matches text to stock tickers using case-insensitive regex with word boundaries.
    """
    
    def __init__(self, ticker_dict: Dict[str, List[str]]):
        """
        Initialize stock matcher.
        
        Args:
            ticker_dict: Dictionary mapping ticker symbols to lists of company name variations
                        Example: {"INFY": ["infosys", "infosys ltd"], "TCS": ["tcs", "tata consultancy services"]}
        """
        self.ticker_dict = ticker_dict
        self.patterns = self._build_patterns()
        logger.info(f"Initialized StockMatcher for {len(ticker_dict)} tickers")
    
    def _build_patterns(self) -> Dict[str, List[re.Pattern]]:
        """
        Build regex patterns for each ticker.
        
        Returns:
            Dict mapping ticker to list of compiled regex patterns
        """
        patterns = {}
        
        for ticker, name_variations in self.ticker_dict.items():
            ticker_patterns = []
            
            # Pattern for ticker symbol (with word boundaries)
            ticker_pattern = re.compile(
                r'\b' + re.escape(ticker.upper()) + r'\b',
                re.IGNORECASE
            )
            ticker_patterns.append(ticker_pattern)
            
            # Patterns for company name variations
            for name in name_variations:
                # Use word boundaries, but allow for common variations
                name_pattern = re.compile(
                    r'\b' + re.escape(name.lower()) + r'\b',
                    re.IGNORECASE
                )
                ticker_patterns.append(name_pattern)
            
            patterns[ticker] = ticker_patterns
        
        return patterns
    
    def find_mentions(self, text: str) -> Set[str]:
        """
        Find all stock tickers mentioned in text.
        
        Args:
            text: Input text to search
            
        Returns:
            Set of ticker symbols found in text
        """
        if not text or len(text.strip()) < 10:
            return set()
        
        mentioned_tickers = set()
        
        for ticker, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    mentioned_tickers.add(ticker)
                    break  # Found this ticker, move to next
        
        return mentioned_tickers
    
    def match_record(self, record: Dict) -> List[str]:
        """
        Match a Reddit record to stock tickers.
        
        Args:
            record: Record dict with 'title', 'selftext' (for submissions) or 'body' (for comments)
            
        Returns:
            List of ticker symbols that match this record
        """
        # Construct text based on record type
        if record.get("type") == "submission":
            text = f"{record.get('title', '')} {record.get('selftext', '')}"
        elif record.get("type") == "comment":
            text = record.get("body", "")
        else:
            # Fallback: try to get any text field
            text = (
                record.get("title", "") + " " +
                record.get("selftext", "") + " " +
                record.get("body", "")
            )
        
        return sorted(list(self.find_mentions(text)))
    
    def add_ticker(self, ticker: str, name_variations: List[str]):
        """
        Add a new ticker to the matcher.
        
        Args:
            ticker: Ticker symbol
            name_variations: List of company name variations
        """
        self.ticker_dict[ticker] = name_variations
        self.patterns[ticker] = self._build_patterns_for_ticker(ticker, name_variations)
        logger.info(f"Added ticker: {ticker}")
    
    def _build_patterns_for_ticker(self, ticker: str, name_variations: List[str]) -> List[re.Pattern]:
        """Build patterns for a single ticker."""
        patterns = []
        
        # Ticker pattern
        patterns.append(re.compile(
            r'\b' + re.escape(ticker.upper()) + r'\b',
            re.IGNORECASE
        ))
        
        # Name patterns
        for name in name_variations:
            patterns.append(re.compile(
                r'\b' + re.escape(name.lower()) + r'\b',
                re.IGNORECASE
            ))
        
        return patterns

