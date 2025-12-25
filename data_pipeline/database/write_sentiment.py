import pandas as pd

# Load processed parquet file
df = pd.read_parquet('data/processed/AAPL_tft.parquet')

# Load and aggregate news sentiment
news = pd.read_csv('data/sentiment/AAPL_news_sentiment.csv')
news['date'] = pd.to_datetime(news['timestamp']).dt.date
news_agg = news.groupby('date')['composite_score'].mean()

# Load and aggregate social sentiment
social = pd.read_csv('data/sentiment/AAPL_social_sentiment.csv')
social['date'] = pd.to_datetime(social['timestamp']).dt.date
social_agg = social.groupby('date')['composite_score'].mean()

# Convert df date column to date if needed
df['date'] = pd.to_datetime(df['date']).dt.date

# Map aggregated sentiment to the parquet DataFrame
df['sentiment_news_composite'] = df['date'].map(news_agg)
df['sentiment_social_score'] = df['date'].map(social_agg)

# Save back to parquet
df.to_parquet('data/processed/AAPL_tft.parquet', index=False)