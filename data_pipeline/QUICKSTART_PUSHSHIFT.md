# Quick Start: Pushshift Reddit Sentiment Pipeline

## Installation

1. Install dependencies:
```bash
pip install zstandard transformers torch pandas numpy
```

2. Download FinBERT model (will auto-download on first run):
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
```

## Step 1: Download Pushshift Data

Download monthly dumps from Pushshift (e.g., from Internet Archive or Pushshift mirrors):
- Submissions: `RS_YYYY-MM.zst` (e.g., `RS_2024-01.zst`)
- Comments: `RC_YYYY-MM.zst` (e.g., `RC_2024-01.zst`)

Place in: `data/pushshift/`

## Step 2: Configure Tickers

Edit `configs/ticker_dict.json`:
```json
{
  "AAPL": ["apple", "apple inc"],
  "MSFT": ["microsoft", "msft"],
  "TSLA": ["tesla", "tesla motors"]
}
```

## Step 3: Run Pipeline

**Single file:**
```bash
python scripts/process_pushshift_reddit.py data/pushshift/RS_2024-01.zst
```

**Directory:**
```bash
python scripts/process_pushshift_reddit.py data/pushshift/
```

**Using main script directly:**
```bash
python data_pipeline/pushshift_sentiment_pipeline.py \
    data/pushshift/RS_2024-01.zst \
    --ticker-config configs/ticker_dict.json \
    --output-dir data/sentiment/reddit \
    --device cuda
```

## Step 4: Output

The pipeline generates:
1. **Unified CSV**: `data/sentiment/reddit/reddit_sentiment_daily.csv`
   - All tickers in one file
   - Columns: `date`, `ticker`, `reddit_sentiment_mean`, `reddit_sentiment_std`, `reddit_post_volume`, `reddit_sentiment_delta`

2. **Per-ticker files** (auto-generated): `data/sentiment/{TICKER}_reddit_sentiment.csv`
   - Compatible with existing TFT pipeline

## Step 5: Integration with TFT

The Reddit sentiment features are automatically integrated when you run:
```bash
python ml_service/data/dataset_generator.py
```

The `add_sentiment_features()` function will:
- Look for `{symbol}_reddit_sentiment.csv` in `data/sentiment/`
- Or read from unified `data/sentiment/reddit/reddit_sentiment_daily.csv`
- Add columns: `reddit_sentiment_mean`, `reddit_post_volume`, `reddit_sentiment_delta`

These become **past covariates** in TFT (not future features).

## Example Output

```csv
date,ticker,reddit_sentiment_mean,reddit_sentiment_std,reddit_post_volume,reddit_sentiment_delta
2024-01-15,AAPL,0.23,0.12,54,0.08
2024-01-16,AAPL,0.31,0.15,67,0.08
2024-01-17,AAPL,0.19,0.10,42,-0.12
```

## Performance Tips

- **GPU**: Use `--device cuda` for 10-100x faster FinBERT inference
- **Batch size**: Increase `--batch-size 64` if you have GPU memory
- **Memory**: Pipeline streams files, so memory usage is low
- **Speed**: ~1000 texts/second on GPU, ~100 texts/second on CPU

## Troubleshooting

**"No records processed"**
- Check subreddit filtering (default: r/stocks, r/investing, r/IndianStockMarket, r/SecurityAnalysis)
- Verify ticker dictionary matches your data
- Check text cleaning isn't too aggressive

**"FinBERT download fails"**
- Check internet connection
- Model caches after first download: `~/.cache/huggingface/`

**"Out of memory"**
- Reduce `--batch-size` (try 16 or 8)
- Process files one at a time

## Academic Citation

When using this pipeline in research, cite:

> Reddit data was sourced from Pushshift archival datasets, a widely used research corpus for large-scale social sentiment analysis, enabling reproducible, API-independent sentiment modeling.

## Next Steps

1. Process multiple months of data
2. Merge with price data for TFT training
3. Validate sentiment signals vs. price movements
4. Use in TFT as past covariates

See `data_pipeline/PUSHSHIFT_PIPELINE.md` for full documentation.

