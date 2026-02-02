import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# =====================================================
# PROJECT ROOT (CORRECT FOR YOUR STRUCTURE)
# File is at: Stock-Mate/ml_service/sentiment/merge_sentiment_into_parquet.py
# parents[2] -> Stock-Mate
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =====================================================
# CONFIG
# =====================================================
SENTIMENT_DIR = PROJECT_ROOT / "data" / "sentiment"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

MODEL_NAME = "ProsusAI/finbert"
DEVICE = torch.device("cpu")

# Prevent CPU overload on Windows
torch.set_num_threads(1)

# =====================================================
# SANITY CHECKS
# =====================================================
assert SENTIMENT_DIR.exists(), f"Missing sentiment directory: {SENTIMENT_DIR}"
assert PROCESSED_DIR.exists(), f"Missing processed directory: {PROCESSED_DIR}"

# =====================================================
# LOAD FINBERT
# =====================================================
print("[INFO] Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
print("[INFO] FinBERT loaded successfully.")

# FinBERT label order: [negative, neutral, positive]
NEG_IDX = 0
POS_IDX = 2

# =====================================================
# BATCH SENTIMENT SCORING
# =====================================================
def finbert_batch_score(texts, batch_size=8):
    """
    texts: list[str]
    returns: list[float]
    composite = positive_prob - negative_prob
    """
    scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        composite = probs[:, POS_IDX] - probs[:, NEG_IDX]
        scores.extend(composite.cpu().numpy().tolist())

    return scores

# =====================================================
# NEWS SENTIMENT PROCESSING
# =====================================================
def process_news_sentiment(ticker):
    path = SENTIMENT_DIR / f"{ticker}_news_sentiment.csv"

    if not path.exists():
        print(f"[WARN] Missing news sentiment for {ticker}")
        return pd.Series(dtype=float)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    df["text"] = (
        df["title"].fillna("").astype(str)
        + ". "
        + df["description"].fillna("").astype(str)
    )

    df["composite"] = finbert_batch_score(df["text"].tolist())
    return df.groupby("date")["composite"].mean()

# =====================================================
# SOCIAL SENTIMENT PROCESSING
# =====================================================
def process_social_sentiment(ticker):
    path = SENTIMENT_DIR / f"{ticker}_social_sentiment.csv"

    if not path.exists():
        print(f"[WARN] Missing social sentiment for {ticker}")
        return pd.Series(dtype=float)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["engagement"] = df["engagement"].fillna(1).replace(0, 1)

    df["composite"] = finbert_batch_score(
        df["text"].fillna("").astype(str).tolist()
    )

    df["weighted"] = df["composite"] * df["engagement"]

    return df.groupby("date").apply(
        lambda x: x["weighted"].sum() / x["engagement"].sum()
    )

# =====================================================
# MERGE SENTIMENT INTO TFT PARQUET
# =====================================================
def merge_into_parquet(ticker):
    print(f"\n=== Processing {ticker} ===")

    parquet_path = PROCESSED_DIR / f"{ticker}_tft.parquet"
    print("[INFO] Loading:", parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Missing TFT parquet for {ticker}: {parquet_path}"
        )

    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    news_daily = process_news_sentiment(ticker)
    social_daily = process_social_sentiment(ticker)

    df["sentiment_news_composite"] = df["date"].map(news_daily).fillna(0.0)
    df["sentiment_social_score"] = df["date"].map(social_daily).fillna(0.0)

    df.to_parquet(parquet_path, index=False)

    print(
        f"[DONE] {ticker} | "
        f"News days: {len(news_daily)} | "
        f"Social days: {len(social_daily)}"
    )

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    for ticker in TICKERS:
        merge_into_parquet(ticker)

    print("\n[SUCCESS] Sentiment computation & merge complete.")
