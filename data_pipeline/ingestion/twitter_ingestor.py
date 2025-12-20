import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np

class TwitterIngestor:
    """Fetch Twitter sentiment data"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        
    def fetch_tweets(self, symbol: str, days: int = 7) -> List[Dict]:
        """Fetch tweets mentioning symbol"""
        if self.api_key:
            return self._fetch_from_api(symbol, days)
        else:
            return self._generate_sample_tweets(symbol, days)
    
    def _fetch_from_api(self, symbol: str, days: int) -> List[Dict]:
        """Fetch from real Twitter API (requires credentials)"""
        try:
            import tweepy
            # Twitter API setup would go here
            # For now, use sample data
            return self._generate_sample_tweets(symbol, days)
        except Exception as e:
            print(f"Error fetching tweets: {e}. Using sample data.")
            return self._generate_sample_tweets(symbol, days)
    
    def _generate_sample_tweets(self, symbol: str, days: int) -> List[Dict]:
        """Generate sample tweets for testing"""
        sample_tweets = [
            f"${symbol} is looking bullish! 🚀 Great earnings report today",
            f"Just bought more ${symbol}. Holding long term! 💎",
            f"${symbol} down today but I'm not worried. Strong company",
            f"Ouch, ${symbol} took a hit. Market volatility 📉",
            f"${symbol} beat expectations again! To the moon! 🌙",
            f"Bearish on ${symbol} for now. Waiting for better entry",
            f"${symbol} is my top pick. Undervalued imo",
            f"Why is ${symbol} down so much? Seems like an overreaction",
            f"Sold my ${symbol} position today. Taking profits",
            f"${symbol} mooning! 🚀📈 Love this stock!"
        ]
        
        tweets = []
        for day_offset in range(days):
            for i in range(5):  # 5 tweets per day
                date = datetime.now() - timedelta(days=day_offset, hours=i)
                tweets.append({
                    "symbol": symbol,
                    "timestamp": date,
                    "text": sample_tweets[(day_offset * 5 + i) % len(sample_tweets)],
                    "author": f"trader_{i}_{day_offset}",
                    "engagement": np.random.randint(10, 500)
                })
        return tweets