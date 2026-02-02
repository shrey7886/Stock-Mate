"""
Complete Sentiment Data Pipeline with Finnhub + NewsAPI
Fetches Finnhub (social sentiment) + NewsAPI (news) sentiment, analyzes with FinBERT,
and integrates into the final dataset for TFT training.

Usage:
    python scripts/sentiment_pipeline_finnhub.py --mode full --finnhub-key YOUR_KEY --newsapi-key YOUR_KEY
    python scripts/sentiment_pipeline_finnhub.py --mode fetch_analyze --finnhub-key YOUR_KEY
    python scripts/sentiment_pipeline_finnhub.py --mode merge
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import requests
import logging
import sys
import argparse
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinnhubCollector:
    """Fetch social sentiment data from Finnhub API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
    
    def fetch_ticker_sentiment(self, ticker: str) -> List[Dict]:
        """
        Fetch social sentiment data for a ticker from Finnhub.
        Returns aggregated sentiment from Reddit, Twitter, and other sources.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of sentiment dictionaries with daily aggregations
        """
        try:
            # Finnhub social sentiment endpoint
            url = f"{self.base_url}/stock/social-sentiment"
            params = {
                'symbol': ticker,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Finnhub API error for {ticker}: {response.status_code}")
                return []
            
            data = response.json()
            
            # Finnhub provides sentiment data broken down by source
            results = []
            
            # Reddit sentiment
            if 'reddit' in data and data['reddit']:
                for entry in data['reddit']:
                    results.append({
                        'ticker': ticker,
                        'date': datetime.fromtimestamp(entry['atTime']).date(),
                        'text': f"Reddit mentions: {entry['mention']} posts, Score: {entry['score']}",
                        'source': 'finnhub_reddit',
                        'sentiment_score': entry['score'],
                        'mention_count': entry['mention'],
                        'positive_mention': entry.get('positiveMention', 0),
                        'negative_mention': entry.get('negativeMention', 0)
                    })
            
            # Twitter sentiment
            if 'twitter' in data and data['twitter']:
                for entry in data['twitter']:
                    results.append({
                        'ticker': ticker,
                        'date': datetime.fromtimestamp(entry['atTime']).date(),
                        'text': f"Twitter mentions: {entry['mention']} tweets, Score: {entry['score']}",
                        'source': 'finnhub_twitter',
                        'sentiment_score': entry['score'],
                        'mention_count': entry['mention'],
                        'positive_mention': entry.get('positiveMention', 0),
                        'negative_mention': entry.get('negativeMention', 0)
                    })
            
            logger.info(f"Fetched {len(results)} sentiment entries for {ticker} from Finnhub")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub data for {ticker}: {e}")
            return []
    
    def fetch_all_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch Finnhub social sentiment for multiple tickers"""
        all_data = []
        
        for i, ticker in enumerate(tickers):
            logger.info(f"Processing {ticker} ({i+1}/{len(tickers)})")
            sentiment_data = self.fetch_ticker_sentiment(ticker)
            all_data.extend(sentiment_data)
            
            # Rate limiting: Finnhub free tier allows 60 calls/min
            if (i + 1) % 50 == 0:
                logger.info("Rate limit pause (60 seconds)...")
                import time
                time.sleep(60)
        
        df = pd.DataFrame(all_data)
        logger.info(f"Fetched {len(df)} total sentiment entries from Finnhub")
        
        return df


class AlphaVantageCollector:
    """Fetch sentiment data from Alpha Vantage NEWS_SENTIMENT API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch_ticker_sentiment(self, ticker: str, limit: int = 50) -> List[Dict]:
        """
        Fetch news sentiment for a ticker from Alpha Vantage.
        Returns articles with pre-calculated sentiment scores.
        
        Args:
            ticker: Stock ticker symbol
            limit: Number of articles to fetch (max 1000)
            
        Returns:
            List of article dictionaries with sentiment scores
        """
        if not self.api_key:
            logger.warning("Alpha Vantage key not provided, skipping Alpha Vantage collection")
            return []
        
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'limit': limit,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"Alpha Vantage API error for {ticker}: {response.status_code}")
                return []
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                logger.warning(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
                return []
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return []
            
            feed = data.get('feed', [])
            
            results = []
            for article in feed:
                # Find ticker-specific sentiment
                ticker_sentiment = None
                for ts in article.get('ticker_sentiment', []):
                    if ts['ticker'] == ticker:
                        ticker_sentiment = ts
                        break
                
                if ticker_sentiment:
                    # Parse published date
                    time_published = article.get('time_published', '')
                    try:
                        # Format: YYYYMMDDTHHMMSS
                        pub_date = datetime.strptime(time_published, '%Y%m%dT%H%M%S').date()
                    except:
                        pub_date = datetime.now().date()
                    
                    results.append({
                        'ticker': ticker,
                        'date': pub_date,
                        'text': f"{article.get('title', '')}. {article.get('summary', '')}",
                        'source': 'alphavantage',
                        'sentiment_label': ticker_sentiment.get('ticker_sentiment_label', 'neutral'),
                        'sentiment_score': float(ticker_sentiment.get('ticker_sentiment_score', 0)),
                        'relevance_score': float(ticker_sentiment.get('relevance_score', 0)),
                        'url': article.get('url', '')
                    })
            
            logger.info(f"Fetched {len(results)} articles for {ticker} from Alpha Vantage")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {ticker}: {e}")
            return []
    
    def fetch_all_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch Alpha Vantage sentiment for multiple tickers"""
        all_data = []
        
        for i, ticker in enumerate(tickers):
            articles = self.fetch_ticker_sentiment(ticker)
            all_data.extend(articles)
            
            # Rate limiting: Alpha Vantage free tier is 500 calls/day, ~1 call/min sustained
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(tickers)} tickers, pausing 12 seconds...")
                import time
                time.sleep(12)  # 5 calls per minute to be safe
        
        df = pd.DataFrame(all_data)
        logger.info(f"Fetched {len(df)} total articles from Alpha Vantage")
        
        return df


class NewsAPICollector:
    """Fetch sentiment data from NewsAPI.org"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
    
    def fetch_ticker_news(self, ticker: str, days_back: int = 7, limit: int = 50) -> List[Dict]:
        """
        Fetch recent news articles for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back
            limit: Number of articles to fetch
            
        Returns:
            List of article dictionaries
        """
        if not self.api_key:
            logger.warning("NewsAPI key not provided, skipping NewsAPI collection")
            return []
        
        try:
            url = f"{self.base_url}/everything"
            
            # Search for company name from ticker_dict for better results
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            params = {
                'q': f'{ticker} stock',
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': min(limit, 100),  # NewsAPI max is 100
                'apiKey': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"NewsAPI error for {ticker}: {response.status_code}")
                return []
            
            data = response.json()
            articles = data.get('articles', [])
            
            logger.info(f"Fetched {len(articles)} articles for {ticker} from NewsAPI")
            
            return [{
                'ticker': ticker,
                'date': pd.to_datetime(article['publishedAt']).date(),
                'text': f"{article['title']}. {article.get('description') or ''}",
                'source': 'newsapi',
                'url': article['url']
            } for article in articles if article.get('title')]
            
        except Exception as e:
            logger.error(f"Error fetching NewsAPI data for {ticker}: {e}")
            return []
    
    def fetch_all_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch NewsAPI data for multiple tickers"""
        all_data = []
        
        for i, ticker in enumerate(tickers):
            articles = self.fetch_ticker_news(ticker)
            all_data.extend(articles)
            
            # Rate limiting: NewsAPI free tier has daily limits
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(tickers)} tickers")
        
        df = pd.DataFrame(all_data)
        logger.info(f"Fetched {len(df)} total articles from NewsAPI")
        
        return df


class SentimentAnalyzer:
    """Analyze sentiment using FinBERT"""
    
    def __init__(self, device: Optional[str] = None, batch_size: int = 32):
        """
        Initialize FinBERT analyzer.
        
        Args:
            device: Device for inference ("cuda", "cpu", or None for auto)
            batch_size: Batch size for processing
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            self.device = device
            if not self.device:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.batch_size = batch_size
            
            logger.info(f"Loading FinBERT model on {self.device}...")
            
            self.model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("FinBERT model loaded successfully")
            
        except ImportError:
            logger.error("transformers package required. Install with: pip install transformers torch")
            sys.exit(1)
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of sentiment dictionaries
        """
        import torch
        
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process results
            for j, pred in enumerate(predictions):
                sentiment_idx = torch.argmax(pred).item()
                sentiment_labels = ['positive', 'negative', 'neutral']
                
                results.append({
                    'sentiment_label': sentiment_labels[sentiment_idx],
                    'sentiment_score': pred[sentiment_idx].item(),
                    'positive_score': pred[0].item(),
                    'negative_score': pred[1].item(),
                    'neutral_score': pred[2].item()
                })
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a dataframe.
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            DataFrame with sentiment columns added
        """
        if len(df) == 0:
            logger.warning("Empty dataframe provided for sentiment analysis")
            return df
        
        logger.info(f"Analyzing sentiment for {len(df)} texts...")
        
        texts = df['text'].tolist()
        sentiments = self.analyze_batch(texts)
        
        # Add sentiment columns
        for key in sentiments[0].keys():
            df[key] = [s[key] for s in sentiments]
        
        logger.info("Sentiment analysis complete")
        
        return df


class SentimentAggregator:
    """Aggregate sentiment data to daily level"""
    
    @staticmethod
    def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment data to daily level per ticker.
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            Aggregated DataFrame
        """
        if len(df) == 0:
            logger.warning("Empty dataframe provided for aggregation")
            return pd.DataFrame()
        
        logger.info("Aggregating sentiment to daily level...")
        
        # For Finnhub data that already has sentiment_score, use it
        # For NewsAPI data, use FinBERT scores
        agg_dict = {
            'sentiment_score': ['mean', 'std', 'count'] if 'sentiment_score' in df.columns else [],
            'positive_score': 'mean',
            'negative_score': 'mean',
            'neutral_score': 'mean',
        }
        
        # Add mention counts if available (from Finnhub)
        if 'mention_count' in df.columns:
            agg_dict['mention_count'] = 'sum'
            agg_dict['positive_mention'] = 'sum'
            agg_dict['negative_mention'] = 'sum'
        
        # Remove empty aggregations
        agg_dict = {k: v for k, v in agg_dict.items() if v}
        
        daily = df.groupby(['ticker', 'date', 'source']).agg(agg_dict).reset_index()
        
        # Flatten column names
        daily.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in daily.columns.values]
        
        logger.info(f"Aggregated to {len(daily)} daily sentiment records")
        
        return daily
    
    @staticmethod
    def merge_sources(finnhub_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge Finnhub and NewsAPI sentiment data.
        
        Args:
            finnhub_df: Aggregated Finnhub sentiment
            news_df: Aggregated NewsAPI sentiment
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging Finnhub and NewsAPI sentiment data...")
        
        # Rename columns to distinguish sources
        if len(finnhub_df) > 0:
            finnhub_df = finnhub_df.copy()
            finnhub_df.columns = [f'fh_{col}' if col not in ['ticker', 'date', 'source'] 
                                   else col for col in finnhub_df.columns]
        
        if len(news_df) > 0:
            news_df = news_df.copy()
            news_df.columns = [f'na_{col}' if col not in ['ticker', 'date', 'source'] 
                               else col for col in news_df.columns]
        
        # Merge on ticker and date
        if len(finnhub_df) > 0 and len(news_df) > 0:
            merged = pd.merge(
                finnhub_df.groupby(['ticker', 'date']).first().reset_index(),
                news_df.groupby(['ticker', 'date']).first().reset_index(),
                on=['ticker', 'date'],
                how='outer',
                suffixes=('_fh', '_na')
            )
        elif len(finnhub_df) > 0:
            merged = finnhub_df.groupby(['ticker', 'date']).first().reset_index()
        elif len(news_df) > 0:
            merged = news_df.groupby(['ticker', 'date']).first().reset_index()
        else:
            merged = pd.DataFrame()
        
        logger.info(f"Merged to {len(merged)} records")
        
        return merged


class SentimentDatasetIntegrator:
    """Integrate sentiment data into final dataset"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.final_dataset_path = project_root / 'data_pipeline' / 'final_dataset.csv'
        self.sentiment_dir = project_root / 'data' / 'sentiment'
        self.sentiment_dir.mkdir(parents=True, exist_ok=True)
    
    def merge_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge sentiment data into final dataset.
        
        Args:
            sentiment_df: Daily aggregated sentiment data
            
        Returns:
            Updated final dataset
        """
        logger.info("Loading final dataset...")
        
        if not self.final_dataset_path.exists():
            logger.error(f"Final dataset not found at {self.final_dataset_path}")
            return pd.DataFrame()
        
        final_df = pd.read_csv(self.final_dataset_path)
        
        # Ensure timestamp is datetime
        final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        logger.info(f"Final dataset shape before merge: {final_df.shape}")
        
        # Merge on ticker (symbol) and date (timestamp)
        merged = pd.merge(
            final_df,
            sentiment_df.rename(columns={'date': 'timestamp', 'ticker': 'symbol'}),
            on=['symbol', 'timestamp'],
            how='left'
        )
        
        logger.info(f"Final dataset shape after merge: {merged.shape}")
        
        return merged
    
    def save_final_dataset(self, df: pd.DataFrame):
        """Save updated final dataset"""
        logger.info(f"Saving updated final dataset to {self.final_dataset_path}")
        df.to_csv(self.final_dataset_path, index=False)
        logger.info("Final dataset saved successfully")


def load_tickers(project_root: Path) -> List[str]:
    """Load ticker symbols from ticker_dict.json"""
    ticker_dict_path = project_root / 'configs' / 'ticker_dict.json'
    
    with open(ticker_dict_path, 'r') as f:
        ticker_dict = json.load(f)
    
    tickers = list(ticker_dict.keys())
    logger.info(f"Loaded {len(tickers)} tickers from ticker_dict.json")
    
    return tickers


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Sentiment Data Pipeline')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['fetch_analyze', 'merge', 'full'],
                       help='Pipeline mode')
    parser.add_argument('--finnhub-key', type=str, help='Finnhub API key')
    parser.add_argument('--newsapi-key', type=str, help='NewsAPI key')
    parser.add_argument('--alphavantage-key', type=str, help='Alpha Vantage API key')
    parser.add_argument('--device', type=str, default=None,
                       help='Device for FinBERT (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    sentiment_dir = project_root / 'data' / 'sentiment'
    sentiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tickers
    tickers = load_tickers(project_root)
    logger.info(f"Starting sentiment pipeline with {len(tickers)} tickers")
    
    if args.mode in ['fetch_analyze', 'full']:
        # Step 1: Fetch data from Finnhub
        finnhub_df = pd.DataFrame()
        if args.finnhub_key:
            finnhub_collector = FinnhubCollector(api_key=args.finnhub_key)
            finnhub_df = finnhub_collector.fetch_all_tickers(tickers)
            
            if len(finnhub_df) > 0:
                finnhub_df.to_csv(sentiment_dir / 'finnhub_raw.csv', index=False)
                logger.info("Finnhub data saved to finnhub_raw.csv")
        else:
            logger.warning("No Finnhub API key provided, skipping Finnhub collection")
        
        # Step 2: Fetch data from NewsAPI
        news_df = pd.DataFrame()
        if args.newsapi_key:
            news_collector = NewsAPICollector(api_key=args.newsapi_key)
            news_df = news_collector.fetch_all_tickers(tickers)
            
            if len(news_df) > 0:
                news_df.to_csv(sentiment_dir / 'newsapi_raw.csv', index=False)
                logger.info("NewsAPI data saved to newsapi_raw.csv")
        else:
            logger.warning("No NewsAPI key provided, skipping NewsAPI collection")
        
        # Step 2b: Fetch data from Alpha Vantage (has pre-calculated sentiment)
        av_df = pd.DataFrame()
        if args.alphavantage_key:
            av_collector = AlphaVantageCollector(api_key=args.alphavantage_key)
            av_df = av_collector.fetch_all_tickers(tickers)
            
            if len(av_df) > 0:
                av_df.to_csv(sentiment_dir / 'alphavantage_raw.csv', index=False)
                logger.info("Alpha Vantage data saved to alphavantage_raw.csv")
        else:
            logger.warning("No Alpha Vantage API key provided, skipping Alpha Vantage collection")
        
        # Step 3: Analyze sentiment with FinBERT (only for NewsAPI data, AV already has scores)
        if len(news_df) > 0 and 'text' in news_df.columns:
            analyzer = SentimentAnalyzer(device=args.device)
            news_df = analyzer.analyze_dataframe(news_df)
            news_df.to_csv(sentiment_dir / 'newsapi_analyzed.csv', index=False)
            logger.info("NewsAPI sentiment analyzed and saved")
        
        # Step 4: Aggregate to daily level
        aggregator = SentimentAggregator()
        
        finnhub_daily = pd.DataFrame()
        if len(finnhub_df) > 0:
            finnhub_daily = aggregator.aggregate_daily(finnhub_df)
        
        news_daily = pd.DataFrame()
        if len(news_df) > 0:
            news_daily = aggregator.aggregate_daily(news_df)
        
        av_daily = pd.DataFrame()
        if len(av_df) > 0:
            av_daily = aggregator.aggregate_daily(av_df)
        
        # Step 5: Merge all sources
        # First merge Finnhub and NewsAPI
        sentiment_daily = aggregator.merge_sources(finnhub_daily, news_daily)
        
        # Then merge with Alpha Vantage if available
        if len(av_daily) > 0:
            if len(sentiment_daily) > 0:
                # Rename AV columns
                av_daily_renamed = av_daily.copy()
                av_daily_renamed.columns = [f'av_{col}' if col not in ['ticker', 'date', 'source'] 
                                             else col for col in av_daily_renamed.columns]
                
                sentiment_daily = pd.merge(
                    sentiment_daily,
                    av_daily_renamed.groupby(['ticker', 'date']).first().reset_index(),
                    on=['ticker', 'date'],
                    how='outer',
                    suffixes=('', '_av')
                )
            else:
                sentiment_daily = av_daily
        
        if len(sentiment_daily) > 0:
            sentiment_daily.to_csv(sentiment_dir / 'sentiment_daily.csv', index=False)
            logger.info(f"Daily sentiment saved to sentiment_daily.csv ({len(sentiment_daily)} records)")
        else:
            logger.error("No sentiment data generated!")
            sys.exit(1)
    
    if args.mode in ['merge', 'full']:
        # Step 6: Integrate into final dataset
        sentiment_daily_path = sentiment_dir / 'sentiment_daily.csv'
        
        if not sentiment_daily_path.exists():
            logger.error("sentiment_daily.csv not found. Run fetch_analyze mode first.")
            sys.exit(1)
        
        sentiment_daily = pd.read_csv(sentiment_daily_path)
        
        integrator = SentimentDatasetIntegrator(project_root)
        final_df = integrator.merge_sentiment(sentiment_daily)
        
        if len(final_df) > 0:
            integrator.save_final_dataset(final_df)
            logger.info("Sentiment data integrated into final dataset successfully!")
        else:
            logger.error("Failed to integrate sentiment data")
            sys.exit(1)
    
    logger.info("Pipeline completed successfully!")


if __name__ == '__main__':
    main()
