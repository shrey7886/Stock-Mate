import pandas as pd
df = pd.read_parquet("data/processed/AAPL_tft.parquet")
print("Sentiment columns present:")
print("  sentiment_news_composite:", df["sentiment_news_composite"].notna().sum(), "non-zero values")
print("  sentiment_social_score:", df["sentiment_social_score"].notna().sum(), "non-zero values")
print("\nFirst 5 rows:")
print(df[["timestamp", "close", "sentiment_news_composite", "sentiment_social_score"]].head())