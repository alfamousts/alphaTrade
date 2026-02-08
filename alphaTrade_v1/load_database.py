#!/usr/bin/env python3
"""
Load data from SQLite database with filtering capabilities.
Extracts hourly price data based on exchange, pair, and time parameters.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sqlite3
from typing import Dict, Optional, List, Tuple

# Import our command line options handler
from command_line_options import parse_arguments, validate_arguments, DataFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseLoader:
    """Load data from SQLite database with filtering capabilities."""

    def __init__(self, data_filter: DataFilter, output_dir: str = './output_train'):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_price_data(self) -> pd.DataFrame:
        """Load hourly price data from SQLite database."""
        logger.info("Loading data from SQLite database...")
        
        db_path = Path(self.data_filter.db_path)
        if not db_path.exists():
            logger.error(f"Database file not found: {db_path}")
            sys.exit(1)

        try:
            conn = sqlite3.connect(str(db_path))
            
            # Build query
            query = '''
                SELECT timestamp, date, open, high, low, close, volume
                FROM hourly_price 
                WHERE exchange = ? AND asset = ?
            '''
            
            params = [self.data_filter.exchange, self.data_filter.pair]
            
            # Add time filter if specified
            time_filter = self.data_filter.get_time_filter_sql()
            if time_filter:
                query += f" AND {time_filter}"
            
            query += " ORDER BY timestamp ASC"
            
            logger.info(f"Query: {query}")
            logger.info(f"Parameters: {params}")
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                logger.error(f"No data found for exchange={self.data_filter.exchange}, pair={self.data_filter.pair}")
                sys.exit(1)
            
            # Convert timestamp from unix (seconds) to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            
            # Sort by timestamp to ensure proper order
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Rename columns to lowercase
            df.columns = df.columns.str.lower()
            
            # Drop the 'date' column if it exists (we use 'timestamp' instead)
            if 'date' in df.columns:
                df = df.drop(columns=['date'])
            
            # Ensure proper data types
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            logger.info(f"Loaded {len(df)} rows")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            sys.exit(1)

    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """Validate data quality and print statistics."""
        if df.empty:
            logger.warning("No data to validate")
            return False

        logger.info(f"\n=== Data Quality Report ===")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values:\n{missing_counts[missing_counts > 0]}")
        else:
            logger.info("No missing values found")

        # Check for duplicates
        duplicates = df.duplicated(subset=['timestamp']).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        else:
            logger.info("No duplicate rows found")

        logger.info("=" * 50)
        return True

    def save_data(self, df: pd.DataFrame):
        """Save data to parquet file."""
        if df.empty:
            logger.warning("No data to save")
            return

        # Create datasets/raw directory
        raw_data_dir = self.output_dir / 'datasets' / 'raw'
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        filename = "hourly_price.parquet"
        filepath = raw_data_dir / filename

        try:
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved {len(df)} rows to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

def main():
    """Main function to load data."""
    args = parse_arguments()
    validate_arguments(args)

    # Set up logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create data filter
    data_filter = DataFilter(args)
    data_filter.print_filter_summary()

    # Initialize database loader
    loader = DatabaseLoader(data_filter, args.output_dir)

    try:
        # Load data
        logger.info("Loading data from SQLite database...")
        df = loader.load_price_data()

        # Validate data
        loader.validate_data_quality(df)

        # Save data
        loader.save_data(df)

        logger.info(f"\nAll data saved to {loader.output_dir}")
        logger.info("Ready for feature_engineering.py")

    except Exception as e:
        logger.error(f"Error in data loading process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
