#!/usr/bin/env python3
"""
scripts/fetch_and_merge_sentiment.py - Generate sentiment data aligned with stock data dates
"""
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict

# -----------------------
# Analyzer implementations
# -----------------------
class NewsAnalyzer:
    def __init__(self):
        self.positive_words = {
            "gain", "profit", "surge", "soar", "rally", "bullish", "outperform",
            "strong", "growth", "success", "advance", "boom", "beat", "exceed",
            "rose", "jump", "positive", "upbeat", "upgrade"
        }
        self.negative_words = {
            "loss", "decline", "crash", "plunge", "bearish", "underperform",
            "weak", "fall", "fell", "down", "slump", "miss", "disappoint",
            "negative", "tumble", "drop"
        }

    def analyze(self, text: str) -> Dict[str, float]:
        text_lower = (text or "").lower()
        words = text_lower.split()
        total = max(1, len(words))
        pos = sum(1 for w in words if w in self.positive_words)
        neg = sum(1 for w in words if w in self.negative_words)
        pos_score = pos / total
        neg_score = neg / total
        neu_score = max(0.0, 1.0 - pos_score - neg_score)
        return {
            "positive": float(pos_score),
            "negative": float(neg_score),
            "neutral": float(neu_score),
            "composite_score": float(pos_score - neg_score)
        }

class SocialMediaAnalyzer:
    def __init__(self):
        self.positive_words = {
            "bullish", "buy", "moon", "gains", "pump", "long", "hold", "diamond", "hands",
            "strong", "love", "great", "awesome", "good", "positive", "up"
        }
        self.negative_words = {
            "bearish", "sell", "crash", "dump", "short", "loss", "panic",
            "weak", "bad", "hate", "down", "lower", "negative"
        }
        self.positive_emojis = ["🚀", "📈", "💰", "💎", "👍", "✅", "🎉"]
        self.negative_emojis = ["📉", "💔", "😞", "❌", "👎", "🔴", "⚠"]

    def analyze(self, text: str) -> Dict[str, float]:
        text = text or ""
        text_lower = text.lower()
        words = text_lower.split()
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        pos_emoji = sum(1 for e in self.positive_emojis if e in text)
        neg_emoji = sum(1 for e in self.negative_emojis if e in text)
        indicators = pos_count + neg_count + pos_emoji + neg_emoji
        if indicators == 0:
            composite = 0.0
        else:
            composite = (pos_count + pos_emoji - neg_count - neg_emoji) / max(1, indicators)
            composite = max(-1.0, min(1.0, composite))
        return {
            "positive_count": int(pos_count + pos_emoji),
            "negative_count": int(neg_count + neg_emoji),
            "composite_score": float(composite)
        }

# -----------------------
# Ingestors (sample data with matching dates)
# -----------------------
def get_stock_date_range() -> tuple:
    """Get the date range from stock data"""
    try:
        df = pd.read_parquet("data/raw/AAPL.parquet")
        df_clean = df.copy()
        df_clean.columns = [c.lower() for c in df_clean.columns]
        if "date" in df_clean.columns:
            dates = pd.to_datetime(df_clean["date"])
            return dates.min(), dates.max()
        elif "timestamp" in df_clean.columns:
            dates = pd.to_datetime(df_clean["timestamp"])
            return dates.min(), dates.max()
        else:
            # Fallback
            return pd.Timestamp("2020-12-01"), pd.Timestamp("2021-02-28")
    except Exception as e:
        print(f"Could not determine date range: {e}. Using default.")
        return pd.Timestamp("2020-12-01"), pd.Timestamp("2021-02-28")

class NewsIngestor:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.sample_titles = [
            "posts strong quarterly earnings",
            "stock rises on new product launch",
            "faces regulatory challenges",
            "announces strategic partnership",
            "reports revenue decline",
            "stock jumps on analyst upgrade",
            "expands into new market",
            "CEO resigns amid controversy",
            "invests in AI technology",
            "beats market expectations"
        ]

    def fetch_news(self, symbol: str) -> List[Dict]:
        out = []
        current = self.start_date
        day_offset = 0
        while current <= self.end_date:
            for i in range(2):  # 2 news per day
                dt = current + timedelta(hours=i*12)
                title = f"{symbol} {self.sample_titles[(day_offset*2 + i) % len(self.sample_titles)]}"
                out.append({
                    "symbol": symbol,
                    "timestamp": dt.isoformat(),
                    "title": title,
                    "description": f"Sample description for {symbol}",
                    "content": f"Full article content about {symbol}",
                    "source": ["Reuters", "Bloomberg", "CNBC"][(day_offset*2 + i) % 3]
                })
            current += timedelta(days=1)
            day_offset += 1
        return out

class TwitterIngestor:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.sample_tweets = [
            "is looking bullish! 🚀 Great earnings report today",
            "Just bought more. Holding long term! 💎",
            "down today but I'm not worried. Strong company",
            "took a hit. Market volatility 📉",
            "beat expectations again! To the moon! 🌙",
            "Bearish for now. Waiting for better entry",
            "my top pick. Undervalued imo",
            "why is it down so much? overreaction",
            "Sold my position today. Taking profits",
            "mooning! 🚀📈 Love this stock!"
        ]

    def fetch_tweets(self, symbol: str) -> List[Dict]:
        out = []
        current = self.start_date
        day_offset = 0
        while current <= self.end_date:
            for i in range(5):  # 5 tweets per day
                dt = current + timedelta(hours=i*4.8)
                text = f"${symbol} " + self.sample_tweets[(day_offset*5 + i) % len(self.sample_tweets)]
                out.append({
                    "symbol": symbol,
                    "timestamp": dt.isoformat(),
                    "text": text,
                    "author": f"user_{i}_{day_offset}",
                    "engagement": int(np.random.randint(0, 500))
                })
            current += timedelta(days=1)
            day_offset += 1
        return out

# -----------------------
# Writer
# -----------------------
class SentimentWriter:
    def __init__(self, storage_dir: str = "data/sentiment"):
        self.storage = Path(storage_dir)
        self.storage.mkdir(parents=True, exist_ok=True)

    def write_news(self, df: pd.DataFrame, symbol: str):
        if df.empty:
            print(f"No news to write for {symbol}")
            return
        p = self.storage / f"{symbol}_news_sentiment.csv"
        df.to_csv(p, index=False)
        print(f"✓ Saved news sentiment -> {p}")

    def write_social(self, df: pd.DataFrame, symbol: str):
        if df.empty:
            print(f"No social to write for {symbol}")
            return
        p = self.storage / f"{symbol}_social_sentiment.csv"
        df.to_csv(p, index=False)
        print(f"✓ Saved social sentiment -> {p}")

# -----------------------
# Orchestrator
# -----------------------
def fetch_and_save_for_symbol(symbol: str, start_date, end_date):
    news_ing = NewsIngestor(start_date, end_date)
    tw_ing = TwitterIngestor(start_date, end_date)
    news_an = NewsAnalyzer()
    social_an = SocialMediaAnalyzer()
    writer = SentimentWriter()

    # News
    articles = news_ing.fetch_news(symbol)
    news_rows = []
    for a in articles:
        text = f"{a.get('title','')} {a.get('description','')}"
        s = news_an.analyze(text)
        news_rows.append({
            "symbol": symbol,
            "timestamp": a["timestamp"],
            "source": a.get("source"),
            "title": a.get("title"),
            "description": a.get("description"),
            "composite_score": s["composite_score"],
            "positive": s["positive"],
            "negative": s["negative"],
            "neutral": s["neutral"]
        })
    news_df = pd.DataFrame(news_rows)
    writer.write_news(news_df, symbol)

    # Social
    tweets = tw_ing.fetch_tweets(symbol)
    tweet_rows = []
    for t in tweets:
        s = social_an.analyze(t["text"])
        tweet_rows.append({
            "symbol": symbol,
            "timestamp": t["timestamp"],
            "author": t["author"],
            "text": t["text"],
            "engagement": t["engagement"],
            "composite_score": s["composite_score"],
            "positive_count": s["positive_count"],
            "negative_count": s["negative_count"]
        })
    social_df = pd.DataFrame(tweet_rows)
    writer.write_social(social_df, symbol)

def main():
    symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
    
    # Get date range from stock data
    start_date, end_date = get_stock_date_range()
    print(f"Generating sentiment for date range: {start_date.date()} to {end_date.date()}\n")
    
    for sym in symbols:
        print(f"Processing {sym}...")
        fetch_and_save_for_symbol(sym, start_date, end_date)
    
    print("\n✓ Done. Sentiment CSVs are in data/sentiment/")

if __name__ == "__main__":
    main()