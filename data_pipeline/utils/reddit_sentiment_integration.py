"""
Reddit Sentiment Integration Utilities
Converts Reddit sentiment daily CSV to per-ticker format for TFT integration.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def convert_reddit_to_per_ticker(
    reddit_csv_path: Path,
    output_dir: Path,
    ticker_column: str = "ticker"
) -> None:
    """
    Convert Reddit sentiment daily CSV (all tickers) to per-ticker files.
    Creates files like: {ticker}_reddit_sentiment.csv
    
    Args:
        reddit_csv_path: Path to reddit_sentiment_daily.csv
        output_dir: Directory to save per-ticker files
        ticker_column: Name of ticker column in CSV
    """
    if not reddit_csv_path.exists():
        logger.warning(f"Reddit sentiment file not found: {reddit_csv_path}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Reading Reddit sentiment from: {reddit_csv_path}")
    df = pd.read_csv(reddit_csv_path)
    
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    # Group by ticker and save
    for ticker in df[ticker_column].unique():
        ticker_df = df[df[ticker_column] == ticker].copy()
        
        # Rename columns to match expected format
        # Map: reddit_sentiment_mean -> composite_score (for compatibility)
        ticker_df = ticker_df.rename(columns={
            "reddit_sentiment_mean": "composite_score",
            "reddit_post_volume": "volume"
        })
        
        # Select relevant columns
        output_cols = ["date", "composite_score", "reddit_sentiment_std", 
                       "volume", "reddit_sentiment_delta"]
        ticker_df = ticker_df[[col for col in output_cols if col in ticker_df.columns]]
        
        # Add timestamp column (for compatibility)
        ticker_df["timestamp"] = pd.to_datetime(ticker_df["date"])
        
        # Save
        output_path = output_dir / f"{ticker}_reddit_sentiment.csv"
        ticker_df.to_csv(output_path, index=False)
        logger.info(f"Saved {ticker} Reddit sentiment: {output_path} ({len(ticker_df)} days)")
    
    logger.info(f"Converted Reddit sentiment for {df[ticker_column].nunique()} tickers")


def merge_reddit_sentiment_to_price_data(
    price_df: pd.DataFrame,
    reddit_csv_path: Path,
    ticker_column: str = "ticker",
    symbol_column: str = "symbol"
) -> pd.DataFrame:
    """
    Merge Reddit sentiment directly into price DataFrame.
    
    Args:
        price_df: DataFrame with price data (must have date/timestamp and symbol columns)
        reddit_csv_path: Path to reddit_sentiment_daily.csv
        ticker_column: Name of ticker column in Reddit CSV
        symbol_column: Name of symbol column in price DataFrame
        
    Returns:
        DataFrame with Reddit sentiment columns added
    """
    if not reddit_csv_path.exists():
        logger.warning(f"Reddit sentiment file not found: {reddit_csv_path}")
        return price_df
    
    # Ensure price_df has date column
    if "date" not in price_df.columns:
        if "timestamp" in price_df.columns:
            price_df["date"] = pd.to_datetime(price_df["timestamp"]).dt.date
        else:
            raise ValueError("price_df must have 'date' or 'timestamp' column")
    else:
        if not isinstance(price_df["date"].iloc[0], pd.Timestamp):
            price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
    
    # Read Reddit sentiment
    reddit_df = pd.read_csv(reddit_csv_path)
    reddit_df["date"] = pd.to_datetime(reddit_df["date"]).dt.date
    
    # Merge on date and symbol/ticker
    merged = price_df.merge(
        reddit_df,
        left_on=["date", symbol_column],
        right_on=["date", ticker_column],
        how="left",
        suffixes=("", "_reddit")
    )
    
    # Fill NaN with 0 or forward fill (cautiously)
    reddit_cols = ["reddit_sentiment_mean", "reddit_post_volume", "reddit_sentiment_delta"]
    for col in reddit_cols:
        if col in merged.columns:
            # Forward fill within each ticker, limit to 3 days
            merged[col] = merged.groupby(symbol_column)[col].ffill(limit=3)
            merged[col] = merged[col].fillna(0.0)
    
    logger.info(f"Merged Reddit sentiment: {merged[reddit_cols[0]].notna().sum()} rows with data")
    
    return merged

