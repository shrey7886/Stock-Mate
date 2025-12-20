"""
Pushshift Reddit Data Processor
Streams and processes Pushshift monthly dumps (.zst files) without loading entire files into memory.
"""

import json
import io
import zstandard as zstd
from pathlib import Path
from typing import Iterator, Dict, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PushshiftProcessor:
    """
    Processes Pushshift Reddit dumps with streaming decompression.
    Handles both submissions and comments from .zst compressed JSON files.
    """
    
    # Target subreddits for financial sentiment analysis
    TARGET_SUBREDDITS = {
        "stocks",
        "investing", 
        "IndianStockMarket",
        "SecurityAnalysis"
    }
    
    def __init__(self, subreddits: Optional[List[str]] = None):
        """
        Initialize processor.
        
        Args:
            subreddits: Optional list of subreddits to filter. 
                       If None, uses default TARGET_SUBREDDITS.
        """
        self.subreddits = set(subreddits) if subreddits else self.TARGET_SUBREDDITS
        logger.info(f"Initialized PushshiftProcessor for subreddits: {self.subreddits}")
    
    def stream_zst_file(self, file_path: Path) -> Iterator[Dict]:
        """
        Stream decompress a .zst file and yield JSON objects line by line.
        
        Args:
            file_path: Path to .zst file
            
        Yields:
            Dict: Parsed JSON object from each line
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Streaming file: {file_path}")
        
        dctx = zstd.ZstdDecompressor()
        processed = 0
        errors = 0
        
        try:
            with open(file_path, 'rb') as fh:
                with dctx.stream_reader(fh) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                    
                    for line in text_stream:
                        if not line.strip():
                            continue
                        
                        try:
                            obj = json.loads(line)
                            processed += 1
                            
                            if processed % 100000 == 0:
                                logger.debug(f"Processed {processed:,} records from {file_path.name}")
                            
                            yield obj
                            
                        except json.JSONDecodeError as e:
                            errors += 1
                            if errors % 1000 == 0:
                                logger.warning(f"JSON decode errors: {errors} (last: {e})")
                            continue
                            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            raise
        
        logger.info(f"Completed {file_path.name}: {processed:,} records, {errors} errors")
    
    def filter_subreddit(self, record: Dict) -> bool:
        """
        Check if record belongs to target subreddits.
        
        Args:
            record: JSON record from Pushshift dump
            
        Returns:
            bool: True if record should be processed
        """
        subreddit = record.get("subreddit", "").lower()
        return subreddit in self.subreddits
    
    def extract_submission(self, record: Dict) -> Optional[Dict]:
        """
        Extract relevant fields from a submission record.
        
        Args:
            record: Raw JSON record
            
        Returns:
            Dict with extracted fields, or None if invalid
        """
        if not self.filter_subreddit(record):
            return None
        
        try:
            created_utc = record.get("created_utc")
            if not created_utc:
                return None
            
            # Convert Unix timestamp to datetime
            try:
                dt = datetime.fromtimestamp(int(created_utc), tz=None)
            except (ValueError, TypeError):
                return None
            
            title = record.get("title", "")
            selftext = record.get("selftext", "")
            
            # Skip deleted/removed posts
            if title in ["[deleted]", "[removed]"] or selftext in ["[deleted]", "[removed]"]:
                return None
            
            return {
                "type": "submission",
                "created_utc": created_utc,
                "datetime": dt,
                "date": dt.date(),
                "subreddit": record.get("subreddit", "").lower(),
                "title": title,
                "selftext": selftext,
                "id": record.get("id", ""),
                "score": record.get("score", 0),
                "num_comments": record.get("num_comments", 0)
            }
            
        except Exception as e:
            logger.debug(f"Error extracting submission: {e}")
            return None
    
    def extract_comment(self, record: Dict) -> Optional[Dict]:
        """
        Extract relevant fields from a comment record.
        
        Args:
            record: Raw JSON record
            
        Returns:
            Dict with extracted fields, or None if invalid
        """
        if not self.filter_subreddit(record):
            return None
        
        try:
            created_utc = record.get("created_utc")
            if not created_utc:
                return None
            
            # Convert Unix timestamp to datetime
            try:
                dt = datetime.fromtimestamp(int(created_utc), tz=None)
            except (ValueError, TypeError):
                return None
            
            body = record.get("body", "")
            
            # Skip deleted/removed comments
            if body in ["[deleted]", "[removed]", ""]:
                return None
            
            return {
                "type": "comment",
                "created_utc": created_utc,
                "datetime": dt,
                "date": dt.date(),
                "subreddit": record.get("subreddit", "").lower(),
                "body": body,
                "id": record.get("id", ""),
                "score": record.get("score", 0),
                "parent_id": record.get("parent_id", "")
            }
            
        except Exception as e:
            logger.debug(f"Error extracting comment: {e}")
            return None
    
    def process_file(self, file_path: Path, record_type: str = "auto") -> Iterator[Dict]:
        """
        Process a Pushshift dump file and yield extracted records.
        
        Args:
            file_path: Path to .zst file
            record_type: "submission", "comment", or "auto" (detect from filename)
            
        Yields:
            Dict: Extracted and filtered records
        """
        # Auto-detect record type from filename
        if record_type == "auto":
            filename_lower = file_path.name.lower()
            if "submission" in filename_lower or "RS_" in filename_lower:
                record_type = "submission"
            elif "comment" in filename_lower or "RC_" in filename_lower:
                record_type = "comment"
            else:
                logger.warning(f"Could not auto-detect record type for {file_path.name}, defaulting to submission")
                record_type = "submission"
        
        extract_func = self.extract_submission if record_type == "submission" else self.extract_comment
        
        for raw_record in self.stream_zst_file(file_path):
            extracted = extract_func(raw_record)
            if extracted:
                yield extracted

