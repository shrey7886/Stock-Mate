# ml_service/data/feature_functions.py

import pandas as pd
import numpy as np
import pandas_ta as ta


# ---------------- PRICE-DERIVED FEATURES ----------------
def add_price_features(df):
    df["returns_pct"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["high_low_spread"] = df["high"] - df["low"]
    df["open_close_spread"] = df["open"] - df["close"]
    df["rolling_mean_7"] = df["close"].rolling(7).mean()
    df["rolling_std_7"] = df["close"].rolling(7).std()
    df["rolling_std_30"] = df["close"].rolling(30).std()
    return df


# ---------------- TECHNICAL INDICATORS ----------------
import pandas_ta as ta

def add_technical_indicators(df):
    # ---- EMA, SMA, RSI ----
    df["EMA_20"] = ta.ema(df["close"], length=20)
    df["EMA_50"] = ta.ema(df["close"], length=50)
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["RSI_14"] = ta.rsi(df["close"], length=14)

    # ---- MACD ----
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_signal"] = macd["MACDs_12_26_9"]
    df["MACD_hist"] = macd["MACDh_12_26_9"]

    # ---- ATR + True Range ----
    df["ATR_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["true_range"] = ta.true_range(df["high"], df["low"], df["close"])

    # ---- BOLLINGER BANDS (Manual clean version) ----
    # Middle band = 20-day SMA
    df["bb_m"] = df["close"].rolling(20).mean()

    # Standard deviation
    df["bb_std"] = df["close"].rolling(20).std()

    # Upper and lower bands
    df["bb_u"] = df["bb_m"] + 2 * df["bb_std"]
    df["bb_l"] = df["bb_m"] - 2 * df["bb_std"]

    # Bandwidth
    df["bollinger_bandwidth"] = (df["bb_u"] - df["bb_l"]) / df["close"]

    return df




# ---------------- VOLATILITY FEATURES ----------------
def add_volatility_features(df):
    df["volatility_20d"] = df["returns_pct"].rolling(20).std()
    df["volatility_60d"] = df["returns_pct"].rolling(60).std()
    return df


# ---------------- SENTIMENT PLACEHOLDERS ----------------
def add_sentiment_placeholders(df):
    df["sentiment_news_composite"] = 0.0
    df["sentiment_social_score"] = 0.0
    return df


# ---------------- CALENDAR FEATURES ----------------
def add_calendar_features(df):
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


# ---------------- TARGETS ----------------
def add_targets(df, horizon=7):
    for i in range(1, horizon + 1):
        df[f"target_future_{i}"] = df["close"].shift(-i)
    return df
