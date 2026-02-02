"""
Scheduled Sentiment Data Updates
Runs sentiment pipeline on a schedule to keep data fresh.

Setup:
    # Windows - Add to Task Scheduler
    # Linux/Mac - Add to crontab:
    #   0 */6 * * * /path/to/python /path/to/schedule_sentiment.py
"""

import schedule
import time
import subprocess
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sentiment_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SentimentScheduler:
    """Schedules sentiment pipeline execution"""
    
    def __init__(self, newsapi_key: str = None):
        self.newsapi_key = newsapi_key
        self.script_path = Path(__file__).parent / 'sentiment_pipeline.py'
        
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
    
    def run_pipeline(self):
        """Execute the sentiment pipeline"""
        try:
            logger.info("Starting sentiment pipeline execution...")
            
            cmd = [
                'python',
                str(self.script_path),
                '--mode', 'full'
            ]
            
            if self.newsapi_key:
                cmd.extend(['--newsapi-key', self.newsapi_key])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Sentiment pipeline completed successfully")
                logger.info(result.stdout)
            else:
                logger.error("❌ Sentiment pipeline failed")
                logger.error(result.stderr)
        
        except Exception as e:
            logger.error(f"Error running pipeline: {e}")
    
    def schedule_daily(self, hour: int = 0, minute: int = 0):
        """Schedule daily execution"""
        time_str = f"{hour:02d}:{minute:02d}"
        schedule.every().day.at(time_str).do(self.run_pipeline)
        logger.info(f"Scheduled daily sentiment update at {time_str}")
    
    def schedule_every_n_hours(self, hours: int = 6):
        """Schedule execution every N hours"""
        schedule.every(hours).hours.do(self.run_pipeline)
        logger.info(f"Scheduled sentiment update every {hours} hours")
    
    def start(self):
        """Start the scheduler"""
        logger.info("Starting sentiment scheduler...")
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("Scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)


def main():
    """Main entry point"""
    
    # Get NewsAPI key from environment
    import os
    newsapi_key = os.getenv('NEWSAPI_KEY')
    
    if not newsapi_key:
        logger.warning("NEWSAPI_KEY environment variable not set. News sentiment will be skipped.")
    
    # Initialize scheduler
    scheduler = SentimentScheduler(newsapi_key=newsapi_key)
    
    # Schedule options:
    # Option 1: Run daily at 6 AM
    scheduler.schedule_daily(hour=6, minute=0)
    
    # Option 2: Run every 6 hours
    # scheduler.schedule_every_n_hours(hours=6)
    
    # Run once at startup
    logger.info("Running initial sentiment pipeline...")
    scheduler.run_pipeline()
    
    # Start scheduler
    scheduler.start()


if __name__ == '__main__':
    main()
