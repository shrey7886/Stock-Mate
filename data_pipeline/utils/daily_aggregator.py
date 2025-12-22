"""
Daily Sentiment Aggregation
Aggregates Reddit sentiment data to daily metrics per stock.
"""

import pandas as pd
from typing import Dict, List
import logging
from datetime import date

logger = logging.getLogger(__name__)


class DailySentimentAggregator:
    """
    Aggregates sentiment data to daily metrics per stock.
    Computes: mean, std, volume, and day-over-day delta.
    """
    
    def __init__(self):
        """Initialize aggregator."""
        pass
    
    def aggregate(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment data to daily metrics per stock.
        
        Args:
            sentiment_df: DataFrame with columns:
                        - date (or datetime)
                        - ticker
                        - sentiment_score
                        - (optional) other metadata
                        
        Returns:
            DataFrame with daily aggregated metrics:
            - date
            - ticker
            - reddit_sentiment_mean
            - reddit_sentiment_std
            - reddit_post_volume
            - reddit_sentiment_delta
        """
        if sentiment_df.empty:
            logger.warning("Empty sentiment DataFrame provided")
            return pd.DataFrame(columns=[
                "date", "ticker", "reddit_sentiment_mean", 
                "reddit_sentiment_std", "reddit_post_volume", "reddit_sentiment_delta"
            ])
        
        df = sentiment_df.copy()
        
        # Ensure date column exists
        if "date" not in df.columns and "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"]).dt.date
        elif "date" not in df.columns and "created_utc" in df.columns:
            df["date"] = pd.to_datetime(df["created_utc"], unit="s").dt.date
        elif "date" in df.columns:
            if not isinstance(df["date"].iloc[0], date):
                df["date"] = pd.to_datetime(df["date"]).dt.date
        
        # Ensure required columns exist
        if "sentiment_score" not in df.columns:
            logger.error("sentiment_score column not found in DataFrame")
            return pd.DataFrame()
        
        if "ticker" not in df.columns:
            logger.error("ticker column not found in DataFrame")
            return pd.DataFrame()
        
        # Group by date and ticker
        daily = df.groupby(["date", "ticker"]).agg({
            "sentiment_score": ["mean", "std", "count"]
        }).reset_index()
        
        # Flatten column names
        daily.columns = ["date", "ticker", "reddit_sentiment_mean", "reddit_sentiment_std", "reddit_post_volume"]
        
        # Fill NaN std with 0 (single post per day)
        daily["reddit_sentiment_std"] = daily["reddit_sentiment_std"].fillna(0.0)
        
        # Sort by date and ticker
        daily = daily.sort_values(["ticker", "date"])
        
        # Calculate day-over-day delta
        daily["reddit_sentiment_delta"] = daily.groupby("ticker")["reddit_sentiment_mean"].diff()
        daily["reddit_sentiment_delta"] = daily["reddit_sentiment_delta"].fillna(0.0)
        
        # Ensure integer volume
        daily["reddit_post_volume"] = daily["reddit_post_volume"].astype(int)
        
        logger.info(f"Aggregated {len(df)} records to {len(daily)} daily entries")
        
        return daily
    
    def merge_with_existing(self, new_daily: pd.DataFrame, existing_path: str = None) -> pd.DataFrame:
        """
        Merge new daily aggregation with existing data.
        
        Args:
            new_daily: New daily aggregated DataFrame
            existing_path: Path to existing CSV file
            
        Returns:
            Merged DataFrame
        """
        if existing_path:
            try:
                existing = pd.read_csv(existing_path, parse_dates=["date"])
                existing["date"] = pd.to_datetime(existing["date"]).dt.date
                
                # Combine and deduplicate
                combined = pd.concat([existing, new_daily], ignore_index=True)
                combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
                combined = combined.sort_values(["ticker", "date"])
                
                logger.info(f"Merged with existing data: {len(existing)} + {len(new_daily)} = {len(combined)} unique entries")
                return combined
            except FileNotFoundError:
                logger.info(f"No existing file at {existing_path}, using new data only")
                return new_daily
        else:
            return new_daily

