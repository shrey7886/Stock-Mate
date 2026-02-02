# SENTIMENT PIPELINE - VISUAL GUIDE

## 🎯 One Command to Complete Everything

```bash
python scripts/sentiment_pipeline.py --mode full
```

That's it! Everything else happens automatically.

---

## 📊 Data Flow Diagram

```
INPUT SOURCES
    │
    ├─→ StockTwits API ─────────────┐
    │   (StockTwits messages)        │
    │   ✓ No API key needed          │
    │   ✓ Real retail sentiment      │
    │                                 │
    └─→ NewsAPI ────────────────────┐│
        (News articles)             ││
        ✓ Free tier available       ││
        ✓ Professional sentiment    ││
                                    ││
                                    ↓↓
                    FETCH & COMBINE DATA
                    ↓
                Raw CSV: sentiment_raw.csv
                (3000-5000 records)

                    ↓

            SENTIMENT ANALYSIS
            (FinBERT - AI Model)
                    ↓
            Analyzed CSV: sentiment_analyzed.csv
            (+ sentiment scores)

                    ↓

            DAILY AGGREGATION
            (Group by date + stock)
                    ↓
            Daily CSV: sentiment_daily.csv
            (45 stocks × ~365 days)

                    ↓

        MERGE WITH PRICE DATA
        (final_dataset.csv)
                    ↓
        OUTPUT: Updated final_dataset.csv
        ✓ Original price columns
        ✓ Technical indicators
        ✓ Sentiment columns (NEW!)
        ✓ Ready for TFT training
```

---

## 🔄 Data Integration Process

### Before Sentiment
```
final_dataset.csv (existing)
├── timestamp
├── symbol
├── open, high, low, close
├── volume
├── technical indicators
│   ├── RSI
│   ├── MACD
│   ├── Bollinger Bands
│   └── Moving Averages
└── target (future price)
```

### After Sentiment Integration
```
final_dataset.csv (updated)
├── timestamp
├── symbol
├── open, high, low, close
├── volume
├── technical indicators (same as before)
├── ★ SENTIMENT COLUMNS (NEW!) ★
│   ├── sentiment_mean      ← PRIMARY SIGNAL
│   ├── sentiment_volume
│   ├── st_sentiment_mean   (StockTwits)
│   ├── st_volume
│   ├── na_sentiment_mean   (NewsAPI)
│   └── na_volume
└── target (future price)
```

---

## ⚡ Quick Start Path (5 minutes)

```
Step 1: Install       (2 min)
┌──────────────────────────────┐
│ pip install -r               │
│ sentiment_requirements.txt    │
└──────┬───────────────────────┘
       │
Step 2: Run Pipeline  (3 min)
┌──────────────────────────────┐
│ python scripts/              │
│ sentiment_pipeline.py         │
│ --mode full                  │
└──────┬───────────────────────┘
       │
       ▼
   ✅ DONE!
   
   Your data is ready for TFT.
```

---

## 📈 Sentiment Interpretation

### Sentiment Score Range: -1.0 to +1.0

```
      NEGATIVE          NEUTRAL          POSITIVE
        │                 │                  │
    -1.0 ─────────────────0─────────────────+1.0
        │                 │                  │
        │                 │                  │
    "Bearish"        "Neutral"          "Bullish"
    Bad news         No clear        Good news
    Sell pressure    direction       Buy pressure

Example:
  sentiment_mean = -0.3  → Slightly bearish (more sales talk)
  sentiment_mean = +0.5  → Very bullish (lots of buying enthusiasm)
  sentiment_mean = 0.0   → Mixed opinions
```

### What Causes Sentiment Changes?

**Positive Sentiment (+):**
- Earnings beat
- New product launch
- Analyst upgrade
- Celebrity endorsement
- Market recovery

**Negative Sentiment (-):**
- Earnings miss
- Regulatory issues
- Executive departure
- Market downturn
- Negative news

---

## 🧠 How Sentiment Features Help TFT

```
Traditional Features           Sentiment Features
────────────────────          ──────────────────
Price data                     What people think
Technical patterns             Market sentiment
Volume changes                 Investor emotion

Together create:
SMARTER PREDICTIONS
   ↓
Better trading signals
   ↓
Higher accuracy
   ↓
Better returns! 📈
```

---

## 📋 Configuration Checklist

Before first run, verify:

```
☐ Python installed (3.7+)
☐ internet connection (for API calls)
☐ GPU available (optional, CPU works)
☐ 500MB free disk space
☐ configs/ticker_dict.json has your 45 stocks
☐ data/raw/ has your price files
☐ dependencies installed: pip install -r sentiment_requirements.txt
☐ Set NEWSAPI_KEY (optional): export NEWSAPI_KEY=your_key
```

---

## 🔍 Output File Structure

After running pipeline:

```
Stock-Mate/
│
├── data/sentiment/                          (NEW folder)
│   ├── sentiment_raw.csv                   Raw data from APIs
│   │   └── Columns: ticker, date, text, source
│   │   └── Size: ~50,000 rows
│   │
│   ├── sentiment_analyzed.csv              With FinBERT scores
│   │   └── Columns: + sentiment_score, positive, negative
│   │   └── Size: ~50,000 rows
│   │
│   └── sentiment_daily.csv                 Daily aggregates
│       └── Columns: date, ticker, sentiment_mean, st_volume, na_volume
│       └── Size: ~45 tickers × 365 days
│
├── data_pipeline/
│   └── final_dataset.csv                   ⭐ UPDATED FILE
│       └── All original columns PLUS new sentiment columns
│       └── Ready for TFT training!
│
├── logs/                                    (if using scheduler)
│   └── sentiment_scheduler.log
│
└── scripts/
    ├── sentiment_pipeline.py                (Main script)
    └── schedule_sentiment.py                (Auto-update script)
```

---

## ⏱️ Timeline for TFT Training

```
Day 1 (You):
  ├─ Install dependencies         (5 min)
  ├─ Run sentiment pipeline       (15 min)
  ├─ Verify output files          (5 min)
  └─ Hand to friend               ✅

Day 1 (Your Friend):
  ├─ Load updated dataset         (1 min)
  ├─ Add sentiment to TFT config  (5 min)
  ├─ Start TFT training           (long, depends on config)
  └─ Get predictions!             🎉
```

---

## 🚨 Common Scenarios

### Scenario 1: Quick Prototype
```
✓ Use StockTwits only (no API key needed)
✓ Run once: python scripts/sentiment_pipeline.py --mode full
✓ Get results in 15 minutes
✓ No ongoing updates needed
```

### Scenario 2: Production System
```
✓ Add NewsAPI key for complete coverage
✓ Set up scheduler: python scripts/schedule_sentiment.py
✓ Run every 6 hours automatically
✓ Always have fresh sentiment data
✓ TFT gets latest market insights
```

### Scenario 3: Research/Paper
```
✓ Run full pipeline once
✓ Keep sentiment_daily.csv for analysis
✓ Document methodology in paper
✓ Show correlation: sentiment vs returns
✓ Publish insights!
```

---

## 📱 Sentiment Columns for TFT Features

### Configure TFT Like This:

```python
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer

# Load data
df = pd.read_csv('data_pipeline/final_dataset.csv')

# Define features
tft_config = {
    'target': 'close',  # Predict close price
    
    # Past covariates (history we know)
    'past_covariates': [
        'sentiment_mean',      # Combined sentiment
        'st_volume',          # Social discussion activity
        'na_volume',          # News coverage frequency
        'RSI_14',             # Technical indicators
        'MACD',
        'volatility_20d',
        'returns_pct'
    ],
    
    # Static features (don't change over time)
    'static_features': ['symbol'],
    
    'time_steps': 30,  # Use 30 days of history
}

# Create TFT model
model = TemporalFusionTransformer(
    input_size=len(tft_config['past_covariates']),
    # ... other parameters
)

# Train
model.fit(df)

# Predict with sentiment information!
predictions = model.predict(df[-30:])
```

---

## 🎁 What You're Delivering

```
┌─────────────────────────────────────┐
│  COMPLETE SENTIMENT SOLUTION        │
├─────────────────────────────────────┤
│                                     │
│  ✅ Production-ready code           │
│  ✅ Real data collection            │
│  ✅ AI-powered analysis             │
│  ✅ Automatic integration           │
│  ✅ Scheduled updates               │
│  ✅ Complete documentation          │
│  ✅ Ready for TFT training          │
│                                     │
│  Value Added:                       │
│  • Retail sentiment (StockTwits)    │
│  • Professional sentiment (News)    │
│  • Financial domain AI (FinBERT)    │
│  • No complex workarounds           │
│  • Zero manual integration          │
│                                     │
└─────────────────────────────────────┘
```

---

## ✅ Pre-Handoff Checklist for Friend

Before handing over, verify:

```
DATA QUALITY
☐ final_dataset.csv has sentiment columns
☐ No all-zero sentiment values
☐ Sentiment ranges from -1 to +1
☐ Temporal coverage looks good (no big gaps)

DOCUMENTATION
☐ Friend has SENTIMENT_QUICK_REFERENCE.md
☐ Friend understands sentiment columns
☐ Friend knows how to use in TFT

OPERATIONAL
☐ All dependencies installed
☐ No API errors in output
☐ Files in correct locations
☐ Scheduler running (if using)
```

---

## 🚀 Success Criteria

You'll know it worked when:

✅ `data_pipeline/final_dataset.csv` updated successfully  
✅ New columns: `sentiment_mean`, `st_volume`, `na_volume`  
✅ Sentiment values range from -1.0 to +1.0  
✅ No error messages in logs  
✅ Your friend successfully loads data in TFT  
✅ TFT trains without issues  
✅ Model accuracy improves with sentiment features  

---

## 🎉 Congratulations!

You've successfully:

1. ✅ Identified the sentiment gap
2. ✅ Researched alternatives
3. ✅ Built optimal solution (StockTwits + NewsAPI)
4. ✅ Implemented production code
5. ✅ Documented everything
6. ✅ Ready for handoff to TFT team

**The sentiment module is COMPLETE and ready for the core TFT training!** 🚀
