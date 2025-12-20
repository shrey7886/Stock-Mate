"""
Pushshift Reddit Sentiment Pipeline
Main orchestrator for processing Pushshift dumps and generating daily sentiment signals.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import argparse
import json

# Import pipeline components
from data_pipeline.ingestion.pushshift_processor import PushshiftProcessor
from data_pipeline.utils.stock_matcher import StockMatcher
from sentiment_service.utils.text_cleaner import RedditTextCleaner
from sentiment_service.inference.finbert_analyzer import FinBERTAnalyzer
from data_pipeline.utils.daily_aggregator import DailySentimentAggregator
from data_pipeline.database.write_sentiment import SentimentWriter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PushshiftSentimentPipeline:
    """
    Complete pipeline for processing Pushshift Reddit data into daily sentiment signals.
    """
    
    def __init__(
        self,
        ticker_dict: Dict[str, List[str]],
        output_dir: str = "data/sentiment/reddit",
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            ticker_dict: Dictionary mapping tickers to company name variations
            output_dir: Directory to save output CSV files
            batch_size: Batch size for FinBERT inference
            device: Device for FinBERT ("cuda", "cpu", or None for auto)
        """
        self.ticker_dict = ticker_dict
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.processor = PushshiftProcessor()
        self.stock_matcher = StockMatcher(ticker_dict)
        self.text_cleaner = RedditTextCleaner(min_words=10)
        self.analyzer = FinBERTAnalyzer(device=device, batch_size=batch_size)
        self.aggregator = DailySentimentAggregator()
        self.writer = SentimentWriter(storage_dir=str(self.output_dir))
        
        logger.info(f"Initialized PushshiftSentimentPipeline for {len(ticker_dict)} tickers")
    
    def process_file(
        self,
        file_path: Path,
        record_type: str = "auto",
        max_records: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Process a single Pushshift dump file.
        
        Args:
            file_path: Path to .zst file
            record_type: "submission", "comment", or "auto"
            max_records: Maximum records to process (None for all)
            
        Returns:
            DataFrame with processed records and sentiment scores
        """
        logger.info(f"Processing file: {file_path}")
        
        records = []
        processed_count = 0
        matched_count = 0
        sentiment_count = 0
        
        try:
            for record in self.processor.process_file(file_path, record_type=record_type):
                if max_records and processed_count >= max_records:
                    break
                
                processed_count += 1
                
                # Step 1: Clean text
                cleaned_text = self.text_cleaner.construct_text(record)
                if not cleaned_text:
                    continue
                
                # Step 2: Match to stocks
                matched_tickers = self.stock_matcher.match_record(record)
                if not matched_tickers:
                    continue
                
                matched_count += 1
                
                # Step 3: Analyze sentiment
                sentiment_result = self.analyzer.analyze_sentiment(cleaned_text)
                sentiment_count += 1
                
                # Step 4: Create records for each matched ticker
                for ticker in matched_tickers:
                    records.append({
                        "date": record["date"],
                        "datetime": record["datetime"],
                        "ticker": ticker,
                        "subreddit": record["subreddit"],
                        "type": record["type"],
                        "text": cleaned_text[:500],  # Truncate for storage
                        "sentiment_score": sentiment_result["sentiment_score"],
                        "positive": sentiment_result["positive"],
                        "negative": sentiment_result["negative"],
                        "neutral": sentiment_result["neutral"],
                        "id": record.get("id", ""),
                        "score": record.get("score", 0)
                    })
                
                if processed_count % 10000 == 0:
                    logger.info(
                        f"Processed: {processed_count:,} | "
                        f"Matched: {matched_count:,} | "
                        f"Sentiment: {sentiment_count:,}"
                    )
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            raise
        
        logger.info(
            f"Completed {file_path.name}: "
            f"{processed_count:,} processed, "
            f"{matched_count:,} matched, "
            f"{sentiment_count:,} analyzed, "
            f"{len(records):,} records created"
        )
        
        return pd.DataFrame(records)
    
    def process_directory(
        self,
        directory: Path,
        pattern: str = "*.zst",
        record_type: str = "auto"
    ) -> pd.DataFrame:
        """
        Process all matching files in a directory.
        
        Args:
            directory: Directory containing .zst files
            pattern: File pattern to match
            record_type: Record type for all files
            
        Returns:
            Combined DataFrame from all files
        """
        files = list(directory.glob(pattern))
        logger.info(f"Found {len(files)} files in {directory}")
        
        all_records = []
        
        for file_path in files:
            try:
                df = self.process_file(file_path, record_type=record_type)
                if not df.empty:
                    all_records.append(df)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        if not all_records:
            logger.warning("No records processed from any files")
            return pd.DataFrame()
        
        combined = pd.concat(all_records, ignore_index=True)
        logger.info(f"Combined {len(combined):,} total records from {len(all_records)} files")
        
        return combined
    
    def run_pipeline(
        self,
        input_path: Path,
        output_filename: str = "reddit_sentiment_daily.csv",
        record_type: str = "auto"
    ) -> pd.DataFrame:
        """
        Run complete pipeline: process files -> aggregate -> save.
        
        Args:
            input_path: Path to .zst file or directory
            output_filename: Output CSV filename
            record_type: Record type
        
        Returns:
            Final daily aggregated DataFrame
        """
        logger.info("=" * 60)
        logger.info("Starting Pushshift Reddit Sentiment Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Process files
        if input_path.is_file():
            raw_df = self.process_file(input_path, record_type=record_type)
        elif input_path.is_dir():
            raw_df = self.process_directory(input_path, record_type=record_type)
        else:
            raise ValueError(f"Invalid input path: {input_path}")
        
        if raw_df.empty:
            logger.warning("No data processed, exiting")
            return pd.DataFrame()
        
        logger.info(f"Processed {len(raw_df):,} raw records")
        
        # Step 2: Daily aggregation
        logger.info("Aggregating to daily metrics...")
        daily_df = self.aggregator.aggregate(raw_df)
        
        if daily_df.empty:
            logger.warning("No daily aggregation possible, exiting")
            return pd.DataFrame()
        
        logger.info(f"Generated {len(daily_df):,} daily entries")
        
        # Step 3: Save results
        output_path = self.output_dir / output_filename
        existing_path = output_path if output_path.exists() else None
        
        # Merge with existing if needed
        if existing_path:
            daily_df = self.aggregator.merge_with_existing(daily_df, str(existing_path))
        
        daily_df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved daily sentiment to: {output_path}")
        
        # Step 4: Summary statistics
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Summary")
        logger.info("=" * 60)
        logger.info(f"Total records processed: {len(raw_df):,}")
        logger.info(f"Daily entries generated: {len(daily_df):,}")
        logger.info(f"Tickers covered: {daily_df['ticker'].nunique()}")
        logger.info(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
        logger.info(f"Output file: {output_path}")
        logger.info("=" * 60)
        
        return daily_df


def load_ticker_dict(config_path: Path) -> Dict[str, List[str]]:
    """Load ticker dictionary from JSON config file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Process Pushshift Reddit dumps into daily sentiment signals"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to .zst file or directory containing .zst files"
    )
    parser.add_argument(
        "--ticker-config",
        type=str,
        default="configs/ticker_dict.json",
        help="Path to ticker dictionary JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sentiment/reddit",
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="reddit_sentiment_daily.csv",
        help="Output CSV filename"
    )
    parser.add_argument(
        "--record-type",
        type=str,
        choices=["auto", "submission", "comment"],
        default="auto",
        help="Record type (auto-detect from filename if 'auto')"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device for FinBERT (auto-detect if not specified)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for FinBERT inference"
    )
    
    args = parser.parse_args()
    
    # Load ticker dictionary
    ticker_config = Path(args.ticker_config)
    if not ticker_config.exists():
        logger.error(f"Ticker config not found: {ticker_config}")
        logger.info("Creating example config file...")
        example_config = {
            "INFY": ["infosys", "infosys ltd"],
            "TCS": ["tcs", "tata consultancy services"],
            "RELIANCE": ["reliance", "reliance industries"]
        }
        ticker_config.parent.mkdir(parents=True, exist_ok=True)
        with open(ticker_config, 'w') as f:
            json.dump(example_config, f, indent=2)
        logger.info(f"Created example config at {ticker_config}")
        logger.info("Please edit it with your tickers and run again")
        return
    
    ticker_dict = load_ticker_dict(ticker_config)
    
    # Initialize and run pipeline
    pipeline = PushshiftSentimentPipeline(
        ticker_dict=ticker_dict,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device
    )
    
    input_path = Path(args.input_path)
    daily_df = pipeline.run_pipeline(
        input_path=input_path,
        output_filename=args.output_filename,
        record_type=args.record_type
    )
    
    if not daily_df.empty:
        logger.info("\n✓ Pipeline completed successfully!")
    else:
        logger.warning("\n⚠ Pipeline completed but no data was generated")


if __name__ == "__main__":
    main()

