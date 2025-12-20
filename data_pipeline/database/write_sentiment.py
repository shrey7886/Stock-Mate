import pandas as pd
from pathlib import Path
from typing import Dict, List
import json

class SentimentWriter:
    """Write sentiment data to storage"""
    
    def __init__(self, storage_dir: str = "data/sentiment"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def write_news_sentiment(self, sentiment_df: pd.DataFrame, symbol: str):
        """Write news sentiment data"""
        if sentiment_df.empty:
            print(f"No news sentiment data for {symbol}")
            return
        
        output_path = self.storage_dir / f"{symbol}_news_sentiment.csv"
        sentiment_df.to_csv(output_path, index=False)
        print(f"✓ Saved news sentiment: {output_path}")
        
    def write_social_sentiment(self, sentiment_df: pd.DataFrame, symbol: str):
        """Write social media sentiment data"""
        if sentiment_df.empty:
            print(f"No social sentiment data for {symbol}")
            return
        
        output_path = self.storage_dir / f"{symbol}_social_sentiment.csv"
        sentiment_df.to_csv(output_path, index=False)
        print(f"✓ Saved social sentiment: {output_path}")
        
    def aggregate_daily_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentiment to daily values"""
        if sentiment_df.empty:
            return pd.DataFrame()
        
        sentiment_df = sentiment_df.copy()
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
        sentiment_df['date'] = sentiment_df['timestamp'].dt.date
        
        daily = sentiment_df.groupby('date').agg({
            'composite_score': 'mean'
        }).reset_index()
        daily.columns = ['date', 'composite_score']
        
        return daily