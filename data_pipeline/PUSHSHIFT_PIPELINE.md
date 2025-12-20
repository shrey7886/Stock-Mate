# Pushshift Reddit Sentiment Pipeline

## Overview

This pipeline processes Pushshift historical Reddit dumps to generate daily, stock-wise sentiment signals suitable for Temporal Fusion Transformer (TFT) input. The pipeline is **completely API-free** and uses publicly available Pushshift archival datasets.

## Academic Justification

**Reddit data was sourced from Pushshift archival datasets, a widely used research corpus for large-scale social sentiment analysis, enabling reproducible, API-independent sentiment modeling.**

Pushshift datasets are:
- Publicly available for research use
- Used extensively in academic research (e.g., Reddit sentiment analysis papers)
- Reproducible and verifiable
- Free from API rate limits or OAuth requirements

## Architecture

```
Pushshift Reddit Dumps (.zst)
        ↓
Subreddit Filtering (r/stocks, r/investing, etc.)
        ↓
Stock Mention Matching (ticker + company name)
        ↓
Text Cleaning (URLs, emojis, normalization)
        ↓
FinBERT Sentiment Scoring (ProsusAI/finbert)
        ↓
Daily Aggregation (mean, std, volume, delta)
        ↓
CSV / Database Storage
```

## Components

### 1. PushshiftProcessor (`data_pipeline/ingestion/pushshift_processor.py`)
- Streams `.zst` files line-by-line (memory efficient)
- Filters by target subreddits
- Extracts submissions and comments
- Handles malformed JSON gracefully

### 2. StockMatcher (`data_pipeline/utils/stock_matcher.py`)
- Case-insensitive regex matching
- Word boundary detection (avoids false positives)
- Supports ticker symbols and company name variations
- Multi-ticker matching per post

### 3. RedditTextCleaner (`sentiment_service/utils/text_cleaner.py`)
- Removes URLs, emojis, excessive punctuation
- Normalizes whitespace
- Preserves financial terminology (important for FinBERT)
- Minimum word count filtering

### 4. FinBERTAnalyzer (`sentiment_service/inference/finbert_analyzer.py`)
- Uses `ProsusAI/finbert` model
- Outputs continuous sentiment score: `positive_prob - negative_prob`
- Batch processing for efficiency
- GPU/CPU auto-detection

### 5. DailySentimentAggregator (`data_pipeline/utils/daily_aggregator.py`)
- Aggregates to daily metrics per stock:
  - `reddit_sentiment_mean`: Mean sentiment score
  - `reddit_sentiment_std`: Standard deviation
  - `reddit_post_volume`: Number of posts/comments
  - `reddit_sentiment_delta`: Day-over-day change

## Usage

### 1. Download Pushshift Dumps

Download monthly dumps from Pushshift:
- Submissions: `RS_YYYY-MM.zst`
- Comments: `RC_YYYY-MM.zst`

Place them in a directory (e.g., `data/pushshift/`).

### 2. Configure Ticker Dictionary

Edit `configs/ticker_dict.json`:

```json
{
  "INFY": ["infosys", "infosys ltd"],
  "TCS": ["tcs", "tata consultancy services"],
  "AAPL": ["apple", "apple inc"]
}
```

### 3. Run Pipeline

**Single file:**
```bash
python data_pipeline/pushshift_sentiment_pipeline.py data/pushshift/RS_2024-01.zst
```

**Directory of files:**
```bash
python data_pipeline/pushshift_sentiment_pipeline.py data/pushshift/ --record-type auto
```

**With options:**
```bash
python data_pipeline/pushshift_sentiment_pipeline.py \
    data/pushshift/RS_2024-01.zst \
    --ticker-config configs/ticker_dict.json \
    --output-dir data/sentiment/reddit \
    --output-filename reddit_sentiment_daily.csv \
    --device cuda \
    --batch-size 64
```

### 4. Output Format

The pipeline generates a CSV file with daily aggregated metrics:

| date       | ticker | reddit_sentiment_mean | reddit_sentiment_std | reddit_post_volume | reddit_sentiment_delta |
|------------|--------|----------------------|---------------------|-------------------|----------------------|
| 2024-01-15 | INFY   | 0.23                 | 0.12                | 54                | 0.08                 |
| 2024-01-16 | INFY   | 0.31                 | 0.15                | 67                | 0.08                 |

## Integration with TFT

These features are used as **observed past covariates** in TFT:

```python
past_covariates = [
    "reddit_sentiment_mean",
    "reddit_post_volume", 
    "reddit_sentiment_delta"
]
```

**Do NOT** use as known future features (they are not known in advance).

### Example Integration

```python
import pandas as pd

# Load sentiment data
sentiment_df = pd.read_csv("data/sentiment/reddit/reddit_sentiment_daily.csv")
sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

# Merge with stock price data
price_df = pd.read_csv("data/raw/AAPL.parquet")
price_df["date"] = pd.to_datetime(price_df["date"])

# Merge on date and ticker
merged = price_df.merge(
    sentiment_df,
    on=["date", "ticker"],  # or "symbol" depending on your schema
    how="left"
)

# Forward fill missing days (cautiously)
merged["reddit_sentiment_mean"] = merged.groupby("ticker")["reddit_sentiment_mean"].ffill(limit=3)
merged["reddit_post_volume"] = merged.groupby("ticker")["reddit_post_volume"].fillna(0)
merged["reddit_sentiment_delta"] = merged.groupby("ticker")["reddit_sentiment_delta"].fillna(0)
```

## Validation

Before training TFT:

1. **Plot sentiment vs price:**
   ```python
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(12, 6))
   plt.plot(merged["date"], merged["reddit_sentiment_mean"], label="Sentiment")
   plt.plot(merged["date"], merged["close"] / merged["close"].iloc[0], label="Price (normalized)")
   plt.legend()
   plt.show()
   ```

2. **Check for constant values:**
   ```python
   print(sentiment_df.groupby("ticker")["reddit_sentiment_mean"].describe())
   ```

3. **Check missing days:**
   ```python
   date_range = pd.date_range(start=sentiment_df["date"].min(), end=sentiment_df["date"].max())
   missing = set(date_range) - set(sentiment_df["date"])
   print(f"Missing days: {len(missing)}")
   ```

## Performance Considerations

- **Memory**: Pipeline streams files line-by-line, so memory usage is minimal
- **Speed**: FinBERT on GPU processes ~1000 texts/second (batch_size=32)
- **Storage**: Processed daily aggregations are small (~MBs per year)

## Troubleshooting

### "File not found" error
- Ensure `.zst` files are downloaded and path is correct
- Check file permissions

### "No records processed"
- Verify subreddit filtering (check `TARGET_SUBREDDITS` in `pushshift_processor.py`)
- Check ticker dictionary matches your data
- Verify text cleaning isn't too aggressive (adjust `min_words`)

### FinBERT model download fails
- Check internet connection
- Verify `transformers` and `huggingface-hub` are installed
- Model will be cached after first download

### Out of memory
- Reduce `batch_size` in FinBERTAnalyzer
- Process files one at a time instead of directory

## Dependencies

Required packages (add to `requirements.txt`):
- `zstandard>=0.21.0` - For .zst decompression
- `transformers>=4.30` - For FinBERT
- `torch>=1.13` - PyTorch backend
- `pandas>=1.3` - Data processing
- `numpy>=1.21` - Numerical operations

## License & Attribution

- Pushshift data: Publicly available research datasets
- FinBERT model: ProsusAI/finbert (Hugging Face)
- Pipeline: Part of Stock-Mate project

## References

- Pushshift: https://github.com/pushshift/api
- FinBERT: https://huggingface.co/ProsusAI/finbert
- Temporal Fusion Transformer: https://arxiv.org/abs/1912.09363

