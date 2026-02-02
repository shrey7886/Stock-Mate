# Sentiment Data Pipeline - Complete Setup

## Overview
This pipeline fetches real sentiment data from **StockTwits** (social) and **NewsAPI** (news), analyzes with **FinBERT**, and integrates into your TFT final dataset.

```
StockTwits (social sentiment) ┐
                               ├─→ FinBERT Analysis ─→ Daily Aggregation ─→ Merge with Prices
NewsAPI (news sentiment)       ┘
```

---

## Installation

### 1. Install Dependencies

```bash
# Core dependencies
pip install pandas numpy requests

# Machine Learning (FinBERT)
pip install torch transformers

# Scheduling (optional, for auto-updates)
pip install schedule
```

### 2. Get NewsAPI Key (Optional but Recommended)

1. Go to https://newsapi.org/register
2. Sign up for free (10,000 requests/month)
3. Copy your API key

---

## Setup & Configuration

### Option A: Quick Start (StockTwits Only)

**No setup needed!** StockTwits doesn't require authentication.

```bash
python scripts/sentiment_pipeline.py --mode full
```

This fetches from:
- ✅ StockTwits (free, no key needed)
- ❌ NewsAPI (skipped without key)

### Option B: Full Setup (StockTwits + NewsAPI)

#### Windows:
```cmd
set NEWSAPI_KEY=your_api_key_here
python scripts/sentiment_pipeline.py --mode full
```

#### Linux/Mac:
```bash
export NEWSAPI_KEY=your_api_key_here
python scripts/sentiment_pipeline.py --mode full
```

---

## Running the Pipeline

### Mode 1: Full Pipeline (Recommended First Run)
Fetches data → Analyzes sentiment → Merges with prices

```bash
python scripts/sentiment_pipeline.py --mode full --newsapi-key YOUR_KEY
```

**Output:**
- `data/sentiment/sentiment_raw.csv` - Raw fetched data
- `data/sentiment/sentiment_analyzed.csv` - With FinBERT scores
- `data/sentiment/sentiment_daily.csv` - Daily aggregated metrics
- `data_pipeline/final_dataset.csv` - **UPDATED with sentiment columns!**

### Mode 2: Fetch & Analyze Only
Just collect and analyze, don't integrate yet

```bash
python scripts/sentiment_pipeline.py --mode fetch_analyze
```

**Output:**
- `data/sentiment/sentiment_raw.csv`
- `data/sentiment/sentiment_analyzed.csv`
- `data/sentiment/sentiment_daily.csv`

### Mode 3: Merge Only
Integrate existing sentiment with prices (after manual edits)

```bash
python scripts/sentiment_pipeline.py --mode merge
```

---

## Sentiment Features Added to Final Dataset

After running the pipeline, these columns are added:

| Column | Description | Range | Interpretation |
|--------|-------------|-------|-----------------|
| `st_sentiment_mean` | Average StockTwits sentiment | -1 to +1 | Social sentiment score |
| `st_volume` | # of StockTwits messages | 0+ | Social discussion volume |
| `na_sentiment_mean` | Average NewsAPI sentiment | -1 to +1 | News sentiment score |
| `na_volume` | # of news articles | 0+ | News coverage frequency |
| `sentiment_mean` | Weighted avg (60% ST, 40% NA) | -1 to +1 | **Use this in TFT** |
| `sentiment_volume` | Total mentions (ST + NA) | 0+ | Combined activity |

**For TFT Training:** Use these as **observed past covariates**:
```python
past_covariates = [
    'sentiment_mean',      # Combined sentiment
    'st_volume',           # Social volume
    'na_volume'            # News volume
]
```

---

## Scheduled Updates (Keep Data Fresh)

### Option 1: Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task:
   - **Name**: `Update Stock Sentiment`
   - **Trigger**: Daily at 6 AM
   - **Action**: 
     ```
     Program: C:\path\to\python.exe
     Arguments: scripts/sentiment_pipeline.py --mode fetch_analyze
     Start in: C:\ML Projects\STOCKMATE\Stock-Mate
     ```

### Option 2: Linux/Mac Cron

```bash
# Edit crontab
crontab -e

# Add this line (runs every 6 hours)
0 */6 * * * cd /path/to/STOCKMATE && /usr/bin/python scripts/sentiment_pipeline.py --mode fetch_analyze
```

### Option 3: Python Scheduler (Cross-Platform)

```bash
# Install schedule
pip install schedule

# Run scheduler
python scripts/schedule_sentiment.py
```

The scheduler will:
- Run immediately at startup
- Run daily at 6 AM
- Log all executions to `logs/sentiment_scheduler.log`

---

## Troubleshooting

### "No module named 'transformers'"
```bash
pip install transformers torch
```

### "NEWSAPI_KEY not found"
Either:
- Set environment variable (see Setup section)
- Run without it (uses StockTwits only): `python scripts/sentiment_pipeline.py --mode full`

### "StockTwits API error 429"
Rate limiting. Wait a few minutes and retry.

### "No records found for some stocks"
This is normal! Not all 45 stocks have daily StockTwits/news discussions. The pipeline fills missing days with 0 sentiment.

### "MemoryError on GPU"
Reduce batch size:
```bash
python scripts/sentiment_pipeline.py --mode full --batch-size 8
```

Or use CPU:
```bash
python scripts/sentiment_pipeline.py --mode full --device cpu
```

---

## Data Quality Notes

### Gaps in Data
- Weekends: No StockTwits data (market closed)
- Small-cap stocks: May have 0 mentions some days
- **Solution**: Pipeline forward-fills up to 5 days

### Sentiment Scale
- `-1.0`: Very negative
- `0.0`: Neutral
- `+1.0`: Very positive

### Weighting
The combined sentiment uses:
- **60% StockTwits** (retail sentiment, real-time)
- **40% NewsAPI** (professional analysis, slower)

You can adjust in the code if needed.

---

## Next Steps

### For Your Friend (TFT Training):

1. **Load the updated dataset:**
   ```python
   import pandas as pd
   df = pd.read_csv('data_pipeline/final_dataset.csv')
   print(df.columns)  # Sentiment columns should be here
   ```

2. **Use sentiment as features:**
   ```python
   past_covariates = [
       'sentiment_mean',
       'st_volume', 
       'na_volume',
       # ... other features
   ]
   ```

3. **Train TFT:**
   ```bash
   python scripts/train_tft.sh
   ```

---

## File Structure

```
Stock-Mate/
├── scripts/
│   ├── sentiment_pipeline.py        ← Main pipeline
│   ├── schedule_sentiment.py         ← Auto-updates
│   └── ...
├── data/
│   └── sentiment/
│       ├── sentiment_raw.csv         ← Raw data from APIs
│       ├── sentiment_analyzed.csv    ← With FinBERT scores
│       └── sentiment_daily.csv       ← Daily metrics
├── data_pipeline/
│   └── final_dataset.csv             ← **UPDATED with sentiment!**
├── configs/
│   └── ticker_dict.json              ← 45 stocks config
└── logs/
    └── sentiment_scheduler.log       ← Scheduler logs
```

---

## Performance

- **Fetch Time**: ~5-10 minutes (depends on API response)
- **Analysis Time**: ~5-15 minutes (depends on GPU)
- **GPU**: NVIDIA GPU processes ~1000 texts/second
- **CPU**: Slower but works fine

---

## Questions?

For issues with:
- **StockTwits**: Check https://stocktwits.com (is API up?)
- **NewsAPI**: Check https://newsapi.org (rate limit?)
- **FinBERT**: Check GPU memory

---

## Summary

✅ **Complete setup for real sentiment data**
✅ **Automatic integration with TFT dataset**
✅ **Scheduled updates for fresh insights**
✅ **Ready for production use**

**Run once:** `python scripts/sentiment_pipeline.py --mode full`

Your final dataset is ready for TFT training! 🚀
