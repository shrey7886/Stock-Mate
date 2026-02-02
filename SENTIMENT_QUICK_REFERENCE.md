# SENTIMENT PIPELINE - QUICK REFERENCE FOR TFT TRAINING

## 🚀 Quick Start (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r sentiment_requirements.txt
```

### Step 2: Run Pipeline
```bash
# Option A: StockTwits only (no setup needed)
python scripts/sentiment_pipeline.py --mode full

# Option B: StockTwits + News (with API key)
set NEWSAPI_KEY=your_api_key
python scripts/sentiment_pipeline.py --mode full
```

### Step 3: Use Sentiment in TFT
The updated `data_pipeline/final_dataset.csv` now includes sentiment columns!

```python
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer

# Load data
df = pd.read_csv('data_pipeline/final_dataset.csv')

# Use sentiment features
past_covariates = [
    'sentiment_mean',      # Combined (60% social, 40% news)
    'st_volume',           # StockTwits discussion volume
    'na_volume'            # News article volume
]

# Configure TFT with sentiment
tft_config = {
    'past_covariates': past_covariates,
    'target': 'close',
    # ... other TFT settings
}
```

---

## 📊 Sentiment Columns in final_dataset.csv

| Column | Meaning |
|--------|---------|
| `sentiment_mean` | **USE THIS** - Combined sentiment (-1 to +1) |
| `sentiment_volume` | Total mentions (ST + news articles) |
| `st_sentiment_mean` | StockTwits sentiment only |
| `st_volume` | Number of StockTwits messages |
| `na_sentiment_mean` | News sentiment only |
| `na_volume` | Number of news articles |

---

## 📅 Keep Data Fresh

### Auto-Update Every 6 Hours
```bash
python scripts/schedule_sentiment.py
```

Or use OS scheduler:
- **Windows**: Task Scheduler
- **Linux/Mac**: Crontab

---

## 🎯 What Sentiment Measures

- **Positive sentiment (+1)**: Market optimism, good news
- **Neutral sentiment (0)**: No clear direction
- **Negative sentiment (-1)**: Market pessimism, bad news

---

## ❓ Common Questions

**Q: Do I need a NewsAPI key?**
A: No, StockTwits works without it. NewsAPI adds news sentiment (optional).

**Q: What if some stocks have no sentiment data?**
A: Pipeline fills gaps automatically (forward fill). Missing = neutral/no discussion.

**Q: Can I use sentiment for prediction?**
A: Yes! Use as observed past covariates (known at prediction time).

**Q: How often should I update?**
A: Daily is good, 6-hourly if you want latest sentiment.

**Q: Will GPU help?**
A: Yes, ~3-5x faster. But CPU works fine too.

---

## 🔗 Data Integration Architecture

```
final_dataset.csv (original) +  sentiment_daily.csv
         ↓                                ↓
         └──────────────┬─────────────────┘
                        ↓
             final_dataset.csv (updated)
             with sentiment columns
                        ↓
                   TFT Training
                        ↓
                   Better Predictions!
```

---

## ⚡ Performance

| Task | Time | Resources |
|------|------|-----------|
| Fetch data | 2-5 min | Network I/O |
| Analyze sentiment | 5-15 min | GPU recommended |
| Integrate dataset | <1 min | CPU |
| **Total** | **~15 min** | **GPU or CPU** |

---

## 📝 Sentiment Features for TFT

### Past Covariates (Recommended)
These are **known at prediction time** (past data):

```python
past_covariates = [
    'sentiment_mean',       # The primary sentiment signal
    'st_volume',           # Social activity (meme stocks peak?)
    'na_volume'            # News coverage (important announcements?)
]
```

### Static Features (Optional)
You could also add:
- `'st_positive_mean'` - Proportion of positive posts
- `'na_sentiment_mean'` - Professional sentiment

### Do NOT Use as Known Future Features
These are **not** known in advance:
- ❌ Don't use future sentiment
- ✅ Use only past sentiment

---

## 🛠️ Customization

### Adjust Sentiment Weights
Edit `scripts/sentiment_pipeline.py`, line ~350:
```python
# Change weights (currently 60% social, 40% news)
merged['sentiment_mean'] = (
    (merged['st_sentiment_mean'] * merged['st_volume'] * 0.7) +  # 70% social
    (merged['na_sentiment_mean'] * merged['na_volume'] * 0.3)    # 30% news
) / (total_volume + 1e-8)
```

### Skip a Stock
Edit `configs/ticker_dict.json` and remove the ticker.

### Change Update Schedule
Edit `scripts/schedule_sentiment.py`, line ~60:
```python
# Change from 6 AM daily to 3 AM daily
scheduler.schedule_daily(hour=3, minute=0)

# Or every 12 hours instead of 6
# scheduler.schedule_every_n_hours(hours=12)
```

---

## ✅ Checklist Before TFT Training

- [ ] Run: `python scripts/sentiment_pipeline.py --mode full`
- [ ] Check `data_pipeline/final_dataset.csv` has sentiment columns
- [ ] Check no errors in `logs/sentiment_scheduler.log` (if using scheduler)
- [ ] Verify sentiment columns not all zeros (data fetched successfully)
- [ ] Set up scheduler if you want auto-updates
- [ ] Share with TFT team and start training!

---

## 📞 Support

**Issue: "No sentiment data found"**
- Check if StockTwits is up: https://stocktwits.com
- Check API rate limits
- Run with `--mode fetch_analyze` first

**Issue: "FinBERT out of memory"**
```bash
python scripts/sentiment_pipeline.py --mode full --device cpu
```

**Issue: "API key not recognized"**
- Verify NEWSAPI_KEY environment variable is set
- Don't need it for StockTwits

---

## 🎉 You're All Set!

Your TFT model now has:
- ✅ Price history
- ✅ Technical indicators
- ✅ **Social sentiment (retail investor)**
- ✅ **News sentiment (professional)**

**Time to train and beat the market!** 🚀
