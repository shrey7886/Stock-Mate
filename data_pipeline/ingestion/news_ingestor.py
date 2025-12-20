import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np

class NewsIngestor:
    """Fetch financial news data"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        
    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Fetch news articles for a symbol"""
        if self.api_key:
            return self._fetch_from_api(symbol, days)
        else:
            # Use sample data for development
            return self._generate_sample_news(symbol, days)
    
    def _fetch_from_api(self, symbol: str, days: int) -> List[Dict]:
        """Fetch from real API (requires API key)"""
        try:
            import requests
            base_url = "https://newsapi.org/v2/everything"
            params = {
                "q": f"{symbol} stock",
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": self.api_key
            }
            
            response = requests.get(base_url, params=params)
            articles = response.json().get("articles", [])
            
            news_data = []
            for article in articles:
                news_data.append({
                    "symbol": symbol,
                    "timestamp": pd.to_datetime(article["publishedAt"]),
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "content": article.get("content", ""),
                    "source": article["source"].get("name", "")
                })
            return news_data
        except Exception as e:
            print(f"Error fetching from API: {e}. Using sample data.")
            return self._generate_sample_news(symbol, days)
    
    def _generate_sample_news(self, symbol: str, days: int) -> List[Dict]:
        """Generate sample news for testing"""
        sample_titles = [
            f"{symbol} posts strong quarterly earnings",
            f"{symbol} stock rises on new product launch",
            f"{symbol} faces regulatory challenges",
            f"{symbol} announces strategic partnership",
            f"{symbol} reports revenue decline",
            f"{symbol} stock jumps on analyst upgrade",
            f"{symbol} expands into new market",
            f"{symbol} CEO resigns amid controversy",
            f"{symbol} invests in AI technology",
            f"{symbol} beats market expectations"
        ]
        
        news_data = []
        for day_offset in range(days):
            for i in range(2):  # 2 news per day
                date = datetime.now() - timedelta(days=day_offset)
                news_data.append({
                    "symbol": symbol,
                    "timestamp": date,
                    "title": sample_titles[(day_offset * 2 + i) % len(sample_titles)],
                    "description": f"Latest update on {symbol} company",
                    "content": f"Full article about {symbol} stock performance",
                    "source": ["Reuters", "Bloomberg", "CNBC"][i % 3]
                })
        return news_data