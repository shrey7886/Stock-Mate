@echo off
REM Replace YOUR_FINNHUB_KEY with your actual Finnhub API key from https://finnhub.io
REM NewsAPI key is already set

python scripts/sentiment_pipeline_finnhub.py ^
  --mode full ^
  --finnhub-key d5t6qqhr01qt62nhki80d5t6qqhr01qt62nhki8g ^
  --newsapi-key 2b1aee230c5747c8bfb7dd01c3ee6532

pause
