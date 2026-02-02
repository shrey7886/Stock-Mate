"""
Complete Sentiment Data Pipeline
Fetches StockTwits (social) + NewsAPI (news) sentiment, analyzes with FinBERT,
and integrates into the final dataset for TFT training.

Usage:
    python scripts/sentiment_pipeline.py --mode fetch_analyze
    python scripts/sentiment_pipeline.py --mode merge
    python scripts/sentiment_pipeline.py --mode full
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


class StockTwitsCollector:
    """Fetch sentiment data from StockTwits API (no API key needed)"""
    
    def __init__(self):
        self.base_url = "https://api.stocktwits.com/api/2"
    
    def fetch_ticker_sentiment(self, ticker: str, limit: int = 30) -> List[Dict]:
        """
        Fetch recent messages for a ticker from StockTwits.
        
        Args:
            ticker: Stock ticker symbol
            limit: Number of messages to fetch
            
        Returns:
            List of message dictionaries
        """
        try:
            url = f"{self.base_url}/streams/symbol/{ticker}.json?limit={limit}"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"StockTwits API error for {ticker}: {response.status_code}")
                return []
            
            data = response.json()
            messages = data.get('messages', [])
            
            logger.info(f"Fetched {len(messages)} messages for {ticker} from StockTwits")
            
            return [{
                'ticker': ticker,
                'date': datetime.fromtimestamp(msg['created_at']).date(),
                'text': msg['body'],
                'source': 'stocktwits',
                'sentiment_label': msg['entities']['sentiment']['basic'] if 'entities' in msg and 'sentiment' in msg['entities'] else 'neutral'
            } for msg in messages]
            
        except Exception as e:
            logger.error(f"Error fetching StockTwits data for {ticker}: {e}")
            return []
    
    def fetch_all_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch StockTwits data for multiple tickers"""
        all_data = []
        
        for ticker in tickers:
            messages = self.fetch_ticker_sentiment(ticker)
            all_data.extend(messages)
        
        df = pd.DataFrame(all_data)
        logger.info(f"Fetched {len(df)} total messages from StockTwits")
        
        return df


class NewsAPICollector:
    """Fetch sentiment data from NewsAPI.org"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
    
    def fetch_ticker_news(self, ticker: str, limit: int = 50) -> List[Dict]:
        """
        Fetch recent news articles for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            limit: Number of articles to fetch
            
        Returns:
            List of article dictionaries
        """
        if not self.api_key:
            logger.warning("NewsAPI key not provided, skipping NewsAPI collection")
            return []
        
        try:
            url = f"{self.base_url}/everything"
            params = {
                'q': f'{ticker} stock',
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': limit,
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
                'text': f"{article['title']}. {article['description'] or ''}",
                'source': 'newsapi',
                'sentiment_label': 'neutral'  # Will be analyzed by FinBERT
            } for article in articles if article['title']]
            
        except Exception as e:
            logger.error(f"Error fetching NewsAPI data for {ticker}: {e}")
            return []
    
    def fetch_all_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch NewsAPI data for multiple tickers"""
        all_data = []
        
        for ticker in tickers:
            articles = self.fetch_ticker_news(ticker)
            all_data.extend(articles)
        
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
        from torch.nn import functional as F
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get probabilities
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            
            # Extract sentiment
            # FinBERT: 0=negative, 1=neutral, 2=positive
            for prob in probs:
                results.append({
                    'negative': float(prob[0]),
                    'neutral': float(prob[1]),
                    'positive': float(prob[2]),
                    'sentiment_score': float(prob[2] - prob[0])  # positive - negative
                })
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a DataFrame.
        
        Args:
            df: DataFrame with text column
            text_column: Name of text column
            
        Returns:
            DataFrame with sentiment columns added
        """
        logger.info(f"Analyzing sentiment for {len(df)} texts...")
        
        texts = df[text_column].fillna('').tolist()
        sentiments = self.analyze_batch(texts)
        
        # Add sentiment columns
        df['negative'] = [s['negative'] for s in sentiments]
        df['neutral'] = [s['neutral'] for s in sentiments]
        df['positive'] = [s['positive'] for s in sentiments]
        df['sentiment_score'] = [s['sentiment_score'] for s in sentiments]
        
        logger.info(f"Sentiment analysis complete")
        
        return df


class SentimentAggregator:
    """Aggregate sentiment to daily level per stock"""
    
    @staticmethod
    def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment to daily metrics per stock.
        
        Args:
            df: DataFrame with sentiment scores
            
        Returns:
            DataFrame with daily aggregated metrics
        """
        logger.info("Aggregating sentiment to daily level...")
        
        aggregated = df.groupby(['date', 'ticker']).agg({
            'sentiment_score': ['mean', 'std'],
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean',
            'text': 'count'  # Volume
        }).reset_index()
        
        # Flatten column names
        aggregated.columns = [
            'date',
            'ticker',
            'sentiment_mean',
            'sentiment_std',
            'positive_mean',
            'negative_mean',
            'neutral_mean',
            'sentiment_volume'
        ]
        
        # Fill NaN std with 0
        aggregated['sentiment_std'] = aggregated['sentiment_std'].fillna(0)
        
        logger.info(f"Generated {len(aggregated)} daily sentiment records")
        
        return aggregated
    
    @staticmethod
    def merge_sources(stocktwits_df: pd.DataFrame, newsapi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge StockTwits and NewsAPI sentiment data.
        
        Args:
            stocktwits_df: Daily StockTwits sentiment
            newsapi_df: Daily NewsAPI sentiment
            
        Returns:
            Merged DataFrame with both sources
        """
        logger.info("Merging StockTwits and NewsAPI sentiment...")
        
        # Rename columns to distinguish sources
        st = stocktwits_df.copy()
        st.columns = ['date', 'ticker', 'st_sentiment_mean', 'st_sentiment_std', 
                      'st_positive_mean', 'st_negative_mean', 'st_neutral_mean', 'st_volume']
        
        na = newsapi_df.copy()
        na.columns = ['date', 'ticker', 'na_sentiment_mean', 'na_sentiment_std',
                      'na_positive_mean', 'na_negative_mean', 'na_neutral_mean', 'na_volume']
        
        # Merge
        merged = st.merge(na, on=['date', 'ticker'], how='outer')
        
        # Combine sentiment (weighted average: 60% social, 40% news)
        merged['st_sentiment_mean'] = merged['st_sentiment_mean'].fillna(0)
        merged['na_sentiment_mean'] = merged['na_sentiment_mean'].fillna(0)
        merged['st_volume'] = merged['st_volume'].fillna(0)
        merged['na_volume'] = merged['na_volume'].fillna(0)
        
        # Calculate combined sentiment
        total_volume = merged['st_volume'] + merged['na_volume']
        merged['sentiment_mean'] = (
            (merged['st_sentiment_mean'] * merged['st_volume'] * 0.6) +
            (merged['na_sentiment_mean'] * merged['na_volume'] * 0.4)
        ) / (total_volume + 1e-8)  # Avoid division by zero
        
        # Combined volume
        merged['sentiment_volume'] = merged['st_volume'] + merged['na_volume']
        
        logger.info(f"Merged data: {len(merged)} records")
        
        return merged


class SentimentDatasetIntegrator:
    """Integrate sentiment data into the final TFT dataset"""
    
    def __init__(self, final_dataset_path: str = "data_pipeline/final_dataset.csv"):
        self.final_dataset_path = Path(final_dataset_path)
    
    def load_final_dataset(self) -> pd.DataFrame:
        """Load the final dataset"""
        logger.info(f"Loading final dataset from {self.final_dataset_path}")
        df = pd.read_csv(self.final_dataset_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        return df
    
    def merge_sentiment(self, final_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge sentiment data into final dataset.
        
        Args:
            final_df: Final dataset with prices
            sentiment_df: Merged sentiment data
            
        Returns:
            Dataset with sentiment columns added
        """
        logger.info("Merging sentiment into final dataset...")
        
        # Merge on date and symbol
        merged = final_df.merge(
            sentiment_df,
            left_on=['date', 'symbol'],
            right_on=['date', 'ticker'],
            how='left'
        )
        
        # Drop redundant ticker column
        if 'ticker' in merged.columns:
            merged = merged.drop('ticker', axis=1)
        
        # Forward fill missing sentiment values (up to 5 days)
        sentiment_cols = ['st_sentiment_mean', 'st_volume', 'na_sentiment_mean', 
                         'na_volume', 'sentiment_mean', 'sentiment_volume']
        
        for col in sentiment_cols:
            if col in merged.columns:
                merged[col] = merged.groupby('symbol')[col].fillna(method='ffill', limit=5)
        
        # Fill any remaining NaN with 0 (no sentiment that day)
        for col in sentiment_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        
        logger.info(f"Final dataset shape: {merged.shape}")
        
        return merged
    
    def save_final_dataset(self, df: pd.DataFrame, output_path: Optional[str] = None):
        """Save the integrated dataset"""
        if output_path is None:
            output_path = self.final_dataset_path
        
        logger.info(f"Saving final dataset to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved! Shape: {df.shape}")


def main():
    """Main pipeline execution"""
    
    parser = argparse.ArgumentParser(description='Sentiment Data Pipeline')
    parser.add_argument('--mode', default='full', 
                       choices=['fetch_analyze', 'merge', 'full'],
                       help='Pipeline mode: fetch_analyze (get data), merge (integrate with prices), full (both)')
    parser.add_argument('--newsapi-key', default=None,
                       help='NewsAPI.org API key (optional)')
    parser.add_argument('--device', default=None,
                       choices=['cuda', 'cpu'],
                       help='Device for FinBERT (auto-detect if not specified)')
    parser.add_argument('--output-dir', default='data/sentiment',
                       help='Output directory for sentiment files')
    
    args = parser.parse_args()
    
    # Load ticker dictionary
    with open('configs/ticker_dict.json') as f:
        ticker_dict = json.load(f)
    tickers = list(ticker_dict.keys())
    
    logger.info(f"Starting sentiment pipeline with {len(tickers)} tickers")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================
    # PHASE 1: Fetch & Analyze
    # ============================================
    if args.mode in ['fetch_analyze', 'full']:
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: Fetching & Analyzing Sentiment Data")
        logger.info("="*70 + "\n")
        
        # Fetch StockTwits
        logger.info("Fetching StockTwits data...")
        st_collector = StockTwitsCollector()
        st_raw = st_collector.fetch_all_tickers(tickers)
        
        # Fetch NewsAPI
        logger.info("Fetching NewsAPI data...")
        na_collector = NewsAPICollector(api_key=args.newsapi_key)
        na_raw = na_collector.fetch_all_tickers(tickers)
        
        # Combine raw data
        combined_raw = pd.concat([st_raw, na_raw], ignore_index=True)
        combined_raw.to_csv(output_dir / 'sentiment_raw.csv', index=False)
        logger.info(f"Saved raw sentiment data to sentiment_raw.csv")
        
        # Analyze sentiment with FinBERT
        logger.info("\nAnalyzing sentiment with FinBERT...")
        analyzer = SentimentAnalyzer(device=args.device)
        combined_analyzed = analyzer.analyze_dataframe(combined_raw)
        combined_analyzed.to_csv(output_dir / 'sentiment_analyzed.csv', index=False)
        logger.info(f"Saved analyzed sentiment to sentiment_analyzed.csv")
        
        # Aggregate by source
        st_agg = SentimentAggregator.aggregate_daily(combined_analyzed[combined_analyzed['source'] == 'stocktwits'])
        na_agg = SentimentAggregator.aggregate_daily(combined_analyzed[combined_analyzed['source'] == 'newsapi'])
        
        # Merge sources
        merged_sentiment = SentimentAggregator.merge_sources(st_agg, na_agg)
        merged_sentiment.to_csv(output_dir / 'sentiment_daily.csv', index=False)
        logger.info(f"Saved daily merged sentiment to sentiment_daily.csv")
    
    # ============================================
    # PHASE 2: Merge with Final Dataset
    # ============================================
    if args.mode in ['merge', 'full']:
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: Integrating Sentiment into Final Dataset")
        logger.info("="*70 + "\n")
        
        # Load sentiment if not already loaded
        if args.mode == 'merge':
            merged_sentiment = pd.read_csv(output_dir / 'sentiment_daily.csv')
            merged_sentiment['date'] = pd.to_datetime(merged_sentiment['date']).dt.date
        
        # Integrate
        integrator = SentimentDatasetIntegrator()
        final_df = integrator.load_final_dataset()
        
        # Ensure date is datetime.date for proper merging
        if 'date' not in merged_sentiment.columns:
            merged_sentiment['date'] = pd.to_datetime(merged_sentiment['date']).dt.date
        
        integrated_df = integrator.merge_sentiment(final_df, merged_sentiment)
        
        # Save integrated dataset
        integrator.save_final_dataset(integrated_df)
    
    # ============================================
    # SUMMARY
    # ============================================
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*70)
    
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║                   SENTIMENT PIPELINE COMPLETE!                    ║
╚════════════════════════════════════════════════════════════════════╝

✅ Output Files Generated:
   • {output_dir}/sentiment_raw.csv              - Raw fetched data
   • {output_dir}/sentiment_analyzed.csv         - With FinBERT scores
   • {output_dir}/sentiment_daily.csv            - Daily aggregated metrics
   • data_pipeline/final_dataset.csv             - FINAL (integrated dataset)

✅ Dataset Ready for TFT Training!
   Your friend can now use the updated final_dataset.csv with sentiment features.

📊 Sentiment Features Added:
   • st_sentiment_mean       - StockTwits avg sentiment
   • st_volume               - Number of StockTwits messages
   • na_sentiment_mean       - NewsAPI avg sentiment  
   • na_volume               - Number of news articles
   • sentiment_mean          - Combined weighted sentiment
   • sentiment_volume        - Total sentiment mentions

🚀 Next Steps:
   1. Share data_pipeline/final_dataset.csv with your friend
   2. Use sentiment_mean, st_volume, na_volume as observed past covariates in TFT
   3. Start TFT training!

📝 Notes:
   • To run again with fresh data: python scripts/sentiment_pipeline.py --mode full
   • To update sentiment only: python scripts/sentiment_pipeline.py --mode fetch_analyze
   • To re-integrate with prices: python scripts/sentiment_pipeline.py --mode merge
""")


if __name__ == "__main__":
    main()
