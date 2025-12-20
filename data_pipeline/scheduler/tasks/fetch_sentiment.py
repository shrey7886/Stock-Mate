import pandas as pd
from sentiment_service.inference.analyze_news import NewsAnalyzer
from sentiment_service.inference.analyze_social import SocialMediaAnalyzer
from data_pipeline.ingestion.news_ingestor import NewsIngestor
from data_pipeline.ingestion.twitter_ingestor import TwitterIngestor
from data_pipeline.database.write_sentiment import SentimentWriter
from typing import List

class SentimentFetcher:
    """Fetch and analyze sentiment data"""
    
    def __init__(self):
        self.news_analyzer = NewsAnalyzer(use_finbert=False)  # Use lexicon for speed
        self.social_analyzer = SocialMediaAnalyzer()
        self.news_ingestor = NewsIngestor()
        self.twitter_ingestor = TwitterIngestor()
        self.writer = SentimentWriter()
        
    def fetch_and_analyze_news(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Fetch and analyze news sentiment"""
        print(f"\n📰 Fetching news for {symbol}...")
        articles = self.news_ingestor.fetch_news(symbol, days)
        
        if not articles:
            print(f"No news articles found for {symbol}")
            return pd.DataFrame()
        
        sentiment_data = []
        for article in articles:
            text = f"{article['title']} {article['description']}"
            sentiment = self.news_analyzer.analyze_sentiment(text)
            
            sentiment_data.append({
                "symbol": symbol,
                "timestamp": article['timestamp'],
                "source": article['source'],
                "title": article['title'],
                "composite_score": sentiment['composite_score'],
                "positive": sentiment['positive'],
                "negative": sentiment['negative'],
                "neutral": sentiment['neutral']
            })
        
        df = pd.DataFrame(sentiment_data)
        self.writer.write_news_sentiment(df, symbol)
        return df
    
    def fetch_and_analyze_social(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Fetch and analyze social media sentiment"""
        print(f"🐦 Fetching tweets for {symbol}...")
        tweets = self.twitter_ingestor.fetch_tweets(symbol, days)
        
        if not tweets:
            print(f"No tweets found for {symbol}")
            return pd.DataFrame()
        
        sentiment_data = []
        for tweet in tweets:
            sentiment = self.social_analyzer.analyze_sentiment(tweet['text'])
            
            sentiment_data.append({
                "symbol": symbol,
                "timestamp": tweet['timestamp'],
                "author": tweet['author'],
                "text": tweet['text'],
                "composite_score": sentiment['composite_score'],
                "positive_count": sentiment['positive_count'],
                "negative_count": sentiment['negative_count']
            })
        
        df = pd.DataFrame(sentiment_data)
        self.writer.write_social_sentiment(df, symbol)
        return df
    
    def fetch_all_sentiments(self, symbols: List[str], days: int = 7):
        """Fetch sentiment for all symbols"""
        print(f"\n{'='*60}")
        print(f"Starting sentiment data collection for {len(symbols)} symbols")
        print(f"{'='*60}")
        
        for symbol in symbols:
            try:
                self.fetch_and_analyze_news(symbol, days)
                self.fetch_and_analyze_social(symbol, days)
                print(f"✓ Completed {symbol}")
            except Exception as e:
                print(f"✗ Error processing {symbol}: {e}")
        
        print(f"\n{'='*60}")
        print(f"Sentiment collection complete!")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    fetcher = SentimentFetcher()
    symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
    fetcher.fetch_all_sentiments(symbols, days=7)