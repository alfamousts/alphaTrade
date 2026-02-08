#!/usr/bin/env python3
"""
Command-line options handler for AlphaTrade XGBoost pipeline.
Provides filtering capabilities for exchange, pair, interval, time, and days.
"""

import argparse
import os
from typing import Optional, List
import sys
from datetime import datetime, timedelta

def parse_arguments():
    """Parse command-line arguments for the XGBoost pipeline."""
    parser = argparse.ArgumentParser(
        description="AlphaTrade XGBoost Trading Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python load_database.py --exchange binance --pair BTCUSDT --interval 1h
  python load_database.py --days 365
  python load_database.py --time 1700000000000,1701000000000
        """
    )

    # Exchange filtering
    parser.add_argument(
        '--exchange',
        type=str,
        default='binance',
        help='Exchange name (default: binance)'
    )

    # Pair/Symbol filtering
    parser.add_argument(
        '--pair',
        type=str,
        default='btcusdt.p',
        help='Trading pair (default: btcusdt.p)'
    )

    # Interval filtering (not used for SQLite, but kept for compatibility)
    parser.add_argument(
        '--interval',
        type=str,
        default='1h',
        help='Time interval (default: 1h)'
    )

    # Time range filtering
    parser.add_argument(
        '--time',
        type=str,
        help='Time range as Unix timestamps (ms): start_time,end_time'
    )

    parser.add_argument(
        '--days',
        type=int,
        help='Number of recent days to include'
    )

    # Database path
    parser.add_argument(
        '--db-path',
        type=str,
        default='../sqlite_crypto.db',
        help='Path to SQLite database (default: ../sqlite_crypto.db)'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output_train',
        help='Output directory for processed data (default: ./output_train)'
    )

    # Label configuration
    parser.add_argument(
        '--tp-pct',
        type=float,
        default=0.02,
        help='Take profit percentage (default: 0.02 = 2%%)'
    )
    parser.add_argument(
        '--sl-pct',
        type=float,
        default=0.01,
        help='Stop loss percentage (default: 0.01 = 1%%)'
    )
    parser.add_argument(
        '--horizon-hours',
        type=int,
        default=24,
        help='Label horizon in hours (default: 24 = 1 day)'
    )

    # Model configuration
    parser.add_argument(
        '--model-version',
        type=str,
        default=None,
        help='Model version label for API upload (e.g., alphatrade_btc_v1)'
    )

    # XGBoost training configuration
    parser.add_argument(
        '--xgb-preset',
        type=str,
        default='default',
        help='XGBoost parameter preset (default: default). Example: binance_20260120'
    )
    parser.add_argument(
        '--skip-tuning',
        action='store_true',
        help='Skip hyperparameter tuning and use preset/default parameters'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    # API upload configuration
    parser.add_argument(
        '--upload-to-api',
        action='store_true',
        help='Upload trained models to API server'
    )
    parser.add_argument(
        '--api-domain',
        type=str,
        default='https://api.dragonfortune.ai',
        help='API domain for model upload (default: https://api.dragonfortune.ai)'
    )
    parser.add_argument(
        '--api-version',
        type=str,
        default='v1',
        help='API version (default: v1)'
    )

    return parser.parse_args()

class DataFilter:
    """Handles data filtering based on command-line arguments."""

    def __init__(self, args):
        self.args = args
        self.exchange = args.exchange.lower() if args.exchange else 'binance'
        self.pair = args.pair.lower() if args.pair else 'btcusdt.p'
        self.interval = args.interval
        self.time_range = self._parse_time_range(args.time) if args.time else None
        self.days_filter = args.days
        self.db_path = args.db_path

    def _parse_time_range(self, time_str: str) -> tuple:
        """Parse time range string into (start_time, end_time) tuple."""
        try:
            parts = time_str.split(',')
            if len(parts) == 1:
                start_time = int(parts[0].strip())
                return (start_time, None)
            elif len(parts) == 2:
                start_time = int(parts[0].strip()) if parts[0].strip() else None
                end_time = int(parts[1].strip()) if parts[1].strip() else None
                return (start_time, end_time)
            else:
                raise ValueError("Invalid time range format")
        except ValueError as e:
            print(f"Error parsing time range: {e}")
            print("Expected format: start_time,end_time or just start_time")
            sys.exit(1)

    def get_time_filter_sql(self) -> str:
        """Generate SQL WHERE clause for time filtering."""
        import datetime

        if self.days_filter:
            # Calculate timestamp for N days ago
            now = datetime.datetime.now()
            n_days_ago = now - datetime.timedelta(days=self.days_filter)
            start_timestamp = int(n_days_ago.timestamp())
            return f"timestamp >= {start_timestamp}"

        if self.time_range:
            start_time, end_time = self.time_range
            conditions = []
            if start_time:
                # Convert from milliseconds to seconds if needed
                start_ts = start_time // 1000 if start_time > 1e12 else start_time
                conditions.append(f"timestamp >= {start_ts}")
            if end_time:
                end_ts = end_time // 1000 if end_time > 1e12 else end_time
                conditions.append(f"timestamp <= {end_ts}")
            return " AND ".join(conditions) if conditions else None

        return None

    def print_filter_summary(self):
        """Print summary of active filters."""
        print("=== Active Filters ===")
        print(f"Exchange: {self.exchange}")
        print(f"Pair: {self.pair}")
        print(f"Interval: {self.interval}")
        print(f"Database: {self.db_path}")

        if self.time_range:
            start_time, end_time = self.time_range
            if start_time:
                import datetime
                start_ts = start_time // 1000 if start_time > 1e12 else start_time
                start_dt = datetime.datetime.fromtimestamp(start_ts)
                print(f"Start time: {start_dt} ({start_time})")
            if end_time:
                end_ts = end_time // 1000 if end_time > 1e12 else end_time
                end_dt = datetime.datetime.fromtimestamp(end_ts)
                print(f"End time: {end_dt} ({end_time})")
        if self.days_filter:
            print(f"Days: {self.days_filter}")
        print("=" * 20)

def validate_arguments(args):
    """Validate command-line arguments."""
    # Validate time format
    if args.time:
        try:
            parts = args.time.split(',')
            for part in parts:
                if part.strip():
                    int(part.strip())
        except ValueError:
            print("Error: Invalid time format. Use Unix timestamps in milliseconds.")
            sys.exit(1)

    # Validate days
    if args.days and args.days < 1:
        print("Error: Days must be a positive integer")
        sys.exit(1)

    # Validate percentages
    if args.tp_pct <= 0 or args.tp_pct >= 1:
        print("Error: tp-pct must be between 0 and 1")
        sys.exit(1)

    if args.sl_pct <= 0 or args.sl_pct >= 1:
        print("Error: sl-pct must be between 0 and 1")
        sys.exit(1)

    if args.horizon_hours < 1:
        print("Error: horizon-hours must be >= 1")
        sys.exit(1)

if __name__ == "__main__":
    # Example usage
    args = parse_arguments()
    validate_arguments(args)

    data_filter = DataFilter(args)
    data_filter.print_filter_summary()
