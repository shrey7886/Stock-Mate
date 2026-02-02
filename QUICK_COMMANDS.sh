#!/bin/bash
# Quick Commands for Sentiment Pipeline Setup

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     SENTIMENT PIPELINE - QUICK COMMAND REFERENCE          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================
# INITIAL SETUP (One time)
# ============================================================
echo "🔧 SETUP COMMANDS (Run once):"
echo "════════════════════════════════"
echo ""
echo "1. Install dependencies:"
echo "   pip install -r sentiment_requirements.txt"
echo ""

# ============================================================
# RUNNING THE PIPELINE
# ============================================================
echo "🚀 RUN PIPELINE (Main Command):"
echo "════════════════════════════════"
echo ""
echo "Full pipeline (StockTwits + NewsAPI):"
echo "   python scripts/sentiment_pipeline.py --mode full"
echo ""
echo "StockTwits only (no API key needed):"
echo "   python scripts/sentiment_pipeline.py --mode full"
echo ""
echo "With NewsAPI key:"
echo "   set NEWSAPI_KEY=your_api_key_here"
echo "   python scripts/sentiment_pipeline.py --mode full"
echo ""

# ============================================================
# DIFFERENT MODES
# ============================================================
echo "🔄 PIPELINE MODES:"
echo "════════════════════════════════"
echo ""
echo "Full pipeline (recommended for first run):"
echo "   python scripts/sentiment_pipeline.py --mode full"
echo ""
echo "Fetch & analyze only (don't integrate yet):"
echo "   python scripts/sentiment_pipeline.py --mode fetch_analyze"
echo ""
echo "Merge sentiment with prices (after manual edits):"
echo "   python scripts/sentiment_pipeline.py --mode merge"
echo ""

# ============================================================
# DEVICE OPTIONS
# ============================================================
echo "⚡ DEVICE SELECTION:"
echo "════════════════════════════════"
echo ""
echo "Auto-detect (GPU if available, else CPU):"
echo "   python scripts/sentiment_pipeline.py --mode full"
echo ""
echo "Force GPU (if you have NVIDIA GPU):"
echo "   python scripts/sentiment_pipeline.py --mode full --device cuda"
echo ""
echo "Force CPU (if GPU runs out of memory):"
echo "   python scripts/sentiment_pipeline.py --mode full --device cpu"
echo ""

# ============================================================
# SCHEDULED UPDATES
# ============================================================
echo "📅 AUTOMATIC UPDATES:"
echo "════════════════════════════════"
echo ""
echo "Run scheduler (daily at 6 AM):"
echo "   python scripts/schedule_sentiment.py"
echo ""
echo "Or set up manually:"
echo ""
echo "Linux/Mac cron (every 6 hours):"
echo "   0 */6 * * * cd /path/to/STOCKMATE && python scripts/sentiment_pipeline.py --mode full"
echo ""
echo "Windows Task Scheduler:"
echo "   Program: C:\\path\\to\\python.exe"
echo "   Arguments: scripts/sentiment_pipeline.py --mode full"
echo "   Start in: C:\\path\\to\\STOCKMATE"
echo ""

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
echo "🔐 ENVIRONMENT VARIABLES:"
echo "════════════════════════════════"
echo ""
echo "Windows (Command Prompt):"
echo "   set NEWSAPI_KEY=your_api_key"
echo "   set REDDIT_CLIENT_ID=your_client_id"
echo "   set REDDIT_CLIENT_SECRET=your_secret"
echo ""
echo "Windows (PowerShell):"
echo "   \$env:NEWSAPI_KEY = 'your_api_key'"
echo "   \$env:REDDIT_CLIENT_ID = 'your_client_id'"
echo ""
echo "Linux/Mac:"
echo "   export NEWSAPI_KEY=your_api_key"
echo "   export REDDIT_CLIENT_ID=your_client_id"
echo "   export REDDIT_CLIENT_SECRET=your_secret"
echo ""

# ============================================================
# VERIFICATION
# ============================================================
echo "✅ VERIFICATION:"
echo "════════════════════════════════"
echo ""
echo "Check if sentiment columns were added:"
echo "   python -c \"import pandas as pd; df = pd.read_csv('data_pipeline/final_dataset.csv'); print('Columns:', list(df.columns[-6:])); print('Shape:', df.shape)\""
echo ""
echo "Check sentiment data quality:"
echo "   python -c \"import pandas as pd; df = pd.read_csv('data_pipeline/final_dataset.csv'); print(df['sentiment_mean'].describe())\""
echo ""
echo "Sample sentiment data:"
echo "   python -c \"import pandas as pd; df = pd.read_csv('data_pipeline/final_dataset.csv'); print(df[['timestamp', 'symbol', 'sentiment_mean']].head(10))\""
echo ""

# ============================================================
# TROUBLESHOOTING
# ============================================================
echo "🔧 TROUBLESHOOTING:"
echo "════════════════════════════════"
echo ""
echo "Check for errors:"
echo "   # Check logs in: logs/sentiment_scheduler.log"
echo "   # or run with verbose output:"
echo "   python scripts/sentiment_pipeline.py --mode full 2>&1 | tee pipeline_log.txt"
echo ""
echo "If 'No module' error:"
echo "   pip install -r sentiment_requirements.txt --upgrade"
echo ""
echo "If 'Out of memory' error (GPU):"
echo "   python scripts/sentiment_pipeline.py --mode full --device cpu"
echo ""
echo "If StockTwits API error (429 - rate limit):"
echo "   # Wait 1-2 minutes, then retry"
echo "   python scripts/sentiment_pipeline.py --mode full"
echo ""

# ============================================================
# USAGE IN TFT
# ============================================================
echo "📊 USING IN TFT:"
echo "════════════════════════════════"
echo ""
echo "Load data in Python:"
echo "   import pandas as pd"
echo "   df = pd.read_csv('data_pipeline/final_dataset.csv')"
echo ""
echo "Use sentiment in TFT:"
echo "   past_covariates = ["
echo "       'sentiment_mean',      # Combined sentiment"
echo "       'st_volume',           # StockTwits volume"
echo "       'na_volume',           # News API volume"
echo "       # ... other features"
echo "   ]"
echo ""

# ============================================================
# COMMON WORKFLOWS
# ============================================================
echo "🎯 COMMON WORKFLOWS:"
echo "════════════════════════════════"
echo ""
echo "Workflow 1: First-time setup"
echo "   1. pip install -r sentiment_requirements.txt"
echo "   2. python scripts/sentiment_pipeline.py --mode full"
echo "   3. Share final_dataset.csv with friend"
echo ""
echo "Workflow 2: Regular updates"
echo "   1. python scripts/sentiment_pipeline.py --mode fetch_analyze"
echo "   2. Review sentiment_daily.csv"
echo "   3. python scripts/sentiment_pipeline.py --mode merge"
echo ""
echo "Workflow 3: Continuous monitoring"
echo "   1. python scripts/schedule_sentiment.py"
echo "   2. Runs automatically daily"
echo "   3. Check logs/sentiment_scheduler.log"
echo ""

# ============================================================
# CLEANUP
# ============================================================
echo "🗑️ CLEANUP:"
echo "════════════════════════════════"
echo ""
echo "Remove old sentiment files (keep only latest):"
echo "   rm -rf data/sentiment/sentiment_raw.csv data/sentiment/sentiment_analyzed.csv"
echo "   # Keep: sentiment_daily.csv"
echo ""
echo "Restore final_dataset.csv to original (if needed):"
echo "   git checkout data_pipeline/final_dataset.csv"
echo "   # Then re-run pipeline"
echo ""

# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         For detailed help, see documentation:             ║"
echo "║         • SENTIMENT_QUICK_REFERENCE.md                    ║"
echo "║         • SENTIMENT_PIPELINE_SETUP.md                     ║"
echo "║         • SENTIMENT_VISUAL_GUIDE.md                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
