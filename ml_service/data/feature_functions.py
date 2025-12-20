# ml_service/data/feature_functions.py
"""
Feature engineering helpers — robust to timestamp/date naming and missing sentiment files.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import pandas_ta as ta


# ---------------- Helper ----------------
def ensure_timestamp_and_date(df: pd.DataFrame, timestamp_col: Optional[str] = None) -> pd.DataFrame:
    """
    Ensure the dataframe has:
      - a datetime `timestamp` column (pd.Timestamp)
      - a `date` column (python date) for daily merges
    """
    df = df.copy()

    if timestamp_col:
        if timestamp_col in df.columns:
            df["timestamp"] = pd.to_datetime(df[timestamp_col])
        else:
            raise KeyError(f"Requested timestamp_col '{timestamp_col}' not found in df.columns")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "date" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["date"])
        except Exception:
            df["timestamp"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
    else:
        # Try index
        if pd.api.types.is_datetime64_any_dtype(df.index) or getattr(df.index, "tz", None) is not None:
            df = df.reset_index()
            df["timestamp"] = pd.to_datetime(df["index"])
            df = df.drop(columns=["index"])
        else:
            raise KeyError(
                "No 'timestamp' or 'date' column found and index is not datetime-like. "
                "Please provide a timestamp column named 'timestamp' or 'date'."
            )

    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    return df


# ---------------- PRICE-DERIVED FEATURES ----------------
def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    if "close" not in df.columns:
        raise KeyError("'close' column is required for price features.")
    df = df.copy()
    df["returns_pct"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    if {"high", "low", "open"}.issubset(df.columns):
        df["high_low_spread"] = df["high"] - df["low"]
        df["open_close_spread"] = df["open"] - df["close"]
    df["rolling_mean_7"] = df["close"].rolling(7).mean()
    df["rolling_std_7"] = df["close"].rolling(7).std()
    df["rolling_std_30"] = df["close"].rolling(30).std()
    return df


# ---------------- TECHNICAL INDICATORS ----------------
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if "close" not in df.columns:
        raise KeyError("'close' column is required for technical indicators.")
    df = df.copy()

    # EMA / SMA / RSI
    df["EMA_20"] = ta.ema(df["close"], length=20)
    df["EMA_50"] = ta.ema(df["close"], length=50)
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["RSI_14"] = ta.rsi(df["close"], length=14)

    # MACD
    try:
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        # pandas_ta may return a DataFrame-like mapping
        if isinstance(macd, dict):
            df["MACD"] = macd.get("MACD_12_26_9")
            df["MACD_signal"] = macd.get("MACDs_12_26_9")
            df["MACD_hist"] = macd.get("MACDh_12_26_9")
        else:
            df["MACD"] = macd["MACD_12_26_9"]
            df["MACD_signal"] = macd["MACDs_12_26_9"]
            df["MACD_hist"] = macd["MACDh_12_26_9"]
    except Exception:
        df["MACD"] = np.nan
        df["MACD_signal"] = np.nan
        df["MACD_hist"] = np.nan

    # ATR + True Range
    try:
        df["ATR_14"] = ta.atr(df.get("high"), df.get("low"), df.get("close"), length=14)
        df["true_range"] = ta.true_range(df.get("high"), df.get("low"), df.get("close"))
    except Exception:
        df["ATR_14"] = np.nan
        df["true_range"] = np.nan

    # Bollinger bands
    df["bb_m"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_u"] = df["bb_m"] + 2 * df["bb_std"]
    df["bb_l"] = df["bb_m"] - 2 * df["bb_std"]
    df["bollinger_bandwidth"] = (df["bb_u"] - df["bb_l"]) / df["close"].replace(0, np.nan)

    return df


# ---------------- VOLATILITY FEATURES ----------------
def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    if "returns_pct" not in df.columns:
        df["returns_pct"] = df.get("close").pct_change()
    df["volatility_20d"] = df["returns_pct"].rolling(20).std()
    df["volatility_60d"] = df["returns_pct"].rolling(60).std()
    return df


# ---------------- SENTIMENT FEATURES ----------------
def add_sentiment_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Merge sentiment CSVs from `data/sentiment/{symbol}_*_sentiment.csv` into df.
    If files are missing or malformed, keep zeros.
    """
    df = df.copy()
    df = ensure_timestamp_and_date(df)
    sentiment_dir = Path("data/sentiment")

    # Initialize with zeros
    df["sentiment_news_composite"] = 0.0
    df["sentiment_social_score"] = 0.0
    df["reddit_sentiment_mean"] = 0.0
    df["reddit_post_volume"] = 0.0
    df["reddit_sentiment_delta"] = 0.0

    # News sentiment
    news_path = sentiment_dir / f"{symbol}_news_sentiment.csv"
    if news_path.exists():
        try:
            news = pd.read_csv(news_path)
            # Ensure timestamp is datetime
            news["timestamp"] = pd.to_datetime(news["timestamp"], errors="coerce")
            news["date"] = news["timestamp"].dt.date
            
            # Group by date and take mean
            news_daily = news.groupby("date", as_index=False)["composite_score"].mean()
            
            # Merge: df["date"] is python date, news_daily["date"] is also python date
            df = df.merge(news_daily, on="date", how="left", suffixes=("", "_news"))
            
            # Rename composite_score to sentiment_news_composite if it exists
            if "composite_score_news" in df.columns:
                df["sentiment_news_composite"] = df["composite_score_news"]
                df = df.drop(columns=["composite_score_news"])
            elif "composite_score" in df.columns:
                df["sentiment_news_composite"] = df["composite_score"]
                df = df.drop(columns=["composite_score"])
            
            # Fill NaN with 0
            df["sentiment_news_composite"] = df["sentiment_news_composite"].fillna(0.0)
            print(f"[info] merged news sentiment for {symbol}: {(df['sentiment_news_composite'] != 0.0).sum()} rows with data")
        except Exception as e:
            print(f"[warn] failed to merge news sentiment for {symbol}: {e}")

    # Social sentiment
    social_path = sentiment_dir / f"{symbol}_social_sentiment.csv"
    if social_path.exists():
        try:
            social = pd.read_csv(social_path)
            # Ensure timestamp is datetime
            social["timestamp"] = pd.to_datetime(social["timestamp"], errors="coerce")
            social["date"] = social["timestamp"].dt.date
            
            # Group by date and take mean
            social_daily = social.groupby("date", as_index=False)["composite_score"].mean()
            
            # Merge: df["date"] is python date, social_daily["date"] is also python date
            df = df.merge(social_daily, on="date", how="left", suffixes=("", "_social"))
            
            # Rename composite_score to sentiment_social_score if it exists
            if "composite_score_social" in df.columns:
                df["sentiment_social_score"] = df["composite_score_social"]
                df = df.drop(columns=["composite_score_social"])
            elif "composite_score" in df.columns:
                df["sentiment_social_score"] = df["composite_score"]
                df = df.drop(columns=["composite_score"])
            
            # Fill NaN with 0
            df["sentiment_social_score"] = df["sentiment_social_score"].fillna(0.0)
            print(f"[info] merged social sentiment for {symbol}: {(df['sentiment_social_score'] != 0.0).sum()} rows with data")
        except Exception as e:
            print(f"[warn] failed to merge social sentiment for {symbol}: {e}")

    # Reddit sentiment (from Pushshift pipeline)
    # Option 1: Check for per-ticker file
    reddit_path = sentiment_dir / f"{symbol}_reddit_sentiment.csv"
    # Option 2: Check for unified Reddit CSV
    reddit_unified_path = sentiment_dir / "reddit" / "reddit_sentiment_daily.csv"
    
    if reddit_path.exists():
        try:
            reddit = pd.read_csv(reddit_path)
            reddit["timestamp"] = pd.to_datetime(reddit["timestamp"], errors="coerce")
            reddit["date"] = reddit["timestamp"].dt.date
            
            # Merge reddit_sentiment_mean (or composite_score if renamed)
            if "reddit_sentiment_mean" in reddit.columns:
                reddit_daily = reddit.groupby("date", as_index=False).agg({
                    "reddit_sentiment_mean": "mean",
                    "reddit_post_volume": "sum",
                    "reddit_sentiment_delta": "mean"
                })
            elif "composite_score" in reddit.columns:
                # Legacy format
                reddit_daily = reddit.groupby("date", as_index=False).agg({
                    "composite_score": "mean"
                })
                reddit_daily["reddit_sentiment_mean"] = reddit_daily["composite_score"]
                reddit_daily["reddit_post_volume"] = reddit.get("volume", 0)
                reddit_daily["reddit_sentiment_delta"] = 0.0
            else:
                raise ValueError("Reddit CSV missing expected columns")
            
            df = df.merge(reddit_daily[["date", "reddit_sentiment_mean", "reddit_post_volume", "reddit_sentiment_delta"]], 
                         on="date", how="left", suffixes=("", "_reddit"))
            
            df["reddit_sentiment_mean"] = df["reddit_sentiment_mean"].fillna(0.0)
            df["reddit_post_volume"] = df["reddit_post_volume"].fillna(0.0)
            df["reddit_sentiment_delta"] = df["reddit_sentiment_delta"].fillna(0.0)
            print(f"[info] merged Reddit sentiment for {symbol}: {(df['reddit_sentiment_mean'] != 0.0).sum()} rows with data")
        except Exception as e:
            print(f"[warn] failed to merge Reddit sentiment for {symbol}: {e}")
    elif reddit_unified_path.exists():
        # Try unified Reddit CSV
        try:
            reddit_all = pd.read_csv(reddit_unified_path)
            reddit_all["date"] = pd.to_datetime(reddit_all["date"]).dt.date
            reddit_ticker = reddit_all[reddit_all["ticker"] == symbol].copy()
            
            if not reddit_ticker.empty:
                df = df.merge(
                    reddit_ticker[["date", "reddit_sentiment_mean", "reddit_post_volume", "reddit_sentiment_delta"]],
                    on="date", how="left", suffixes=("", "_reddit")
                )
                df["reddit_sentiment_mean"] = df["reddit_sentiment_mean"].fillna(0.0)
                df["reddit_post_volume"] = df["reddit_post_volume"].fillna(0.0)
                df["reddit_sentiment_delta"] = df["reddit_sentiment_delta"].fillna(0.0)
                print(f"[info] merged Reddit sentiment (unified) for {symbol}: {(df['reddit_sentiment_mean'] != 0.0).sum()} rows with data")
        except Exception as e:
            print(f"[warn] failed to merge unified Reddit sentiment for {symbol}: {e}")

    return df


# ---------------- CALENDAR FEATURES ----------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        df = ensure_timestamp_and_date(df)
    # Use a datetime Series for calendar ops
    dt = pd.to_datetime(df["date"])
    df["day_of_week"] = dt.dt.dayofweek
    try:
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    except Exception:
        df["week_of_year"] = dt.dt.week
    df["month"] = dt.dt.month
    df["is_month_start"] = pd.to_datetime(df.get("timestamp")).dt.is_month_start.astype(int)
    df["is_month_end"] = pd.to_datetime(df.get("timestamp")).dt.is_month_end.astype(int)
    return df


# ---------------- TARGETS ----------------
def add_targets(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    df = df.copy()
    if "close" not in df.columns:
        raise KeyError("'close' column required to create targets.")
    for i in range(1, horizon + 1):
        df[f"target_future_{i}"] = df["close"].shift(-i)
    return df
