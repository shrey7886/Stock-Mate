#!/usr/bin/env python3
"""
Example script to process Pushshift Reddit dumps.
Usage:
    python scripts/process_pushshift_reddit.py data/pushshift/RS_2024-01.zst
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_pipeline.pushshift_sentiment_pipeline import PushshiftSentimentPipeline, load_ticker_dict
import json

def main():
    """Example usage of Pushshift pipeline."""
    
    # Load ticker config
    ticker_config = project_root / "configs" / "ticker_dict.json"
    if not ticker_config.exists():
        print(f"Error: Ticker config not found at {ticker_config}")
        print("Please create it with your ticker dictionary.")
        return 1
    
    ticker_dict = load_ticker_dict(ticker_config)
    print(f"Loaded {len(ticker_dict)} tickers from config")
    
    # Initialize pipeline
    pipeline = PushshiftSentimentPipeline(
        ticker_dict=ticker_dict,
        output_dir="data/sentiment/reddit",
        batch_size=32,
        device=None  # Auto-detect
    )
    
    # Process file or directory
    if len(sys.argv) < 2:
        print("Usage: python scripts/process_pushshift_reddit.py <path_to_zst_file_or_directory>")
        print("\nExample:")
        print("  python scripts/process_pushshift_reddit.py data/pushshift/RS_2024-01.zst")
        print("  python scripts/process_pushshift_reddit.py data/pushshift/")
        return 1
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return 1
    
    # Run pipeline
    try:
        daily_df = pipeline.run_pipeline(
            input_path=input_path,
            output_filename="reddit_sentiment_daily.csv",
            record_type="auto"
        )
        
        if not daily_df.empty:
            print("\n✓ Success! Reddit sentiment data generated.")
            print(f"  Output: data/sentiment/reddit/reddit_sentiment_daily.csv")
            print(f"  Records: {len(daily_df):,}")
            print(f"  Tickers: {daily_df['ticker'].nunique()}")
            print(f"  Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
            
            # Optionally convert to per-ticker format
            from data_pipeline.utils.reddit_sentiment_integration import convert_reddit_to_per_ticker
            convert_reddit_to_per_ticker(
                Path("data/sentiment/reddit/reddit_sentiment_daily.csv"),
                Path("data/sentiment")
            )
            print("\n✓ Also created per-ticker files in data/sentiment/")
        else:
            print("\n⚠ Warning: No data was generated. Check your input files and ticker config.")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

