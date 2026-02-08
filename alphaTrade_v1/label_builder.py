#!/usr/bin/env python3
"""
Label builder for percentage-based TP/SL strategy.
Creates separate labels for LONG and SHORT models.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our command line options handler
from command_line_options import parse_arguments, validate_arguments, DataFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LabelBuilder:
    """Build percentage-based TP/SL labels for LONG and SHORT models."""

    def __init__(self, data_filter: DataFilter, output_dir: str = './output_train',
                 tp_pct: float = 0.02, sl_pct: float = 0.01, horizon_hours: int = 24):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.label_col_long = 'label_long'
        self.label_col_short = 'label_short'
        
        # Label parameters
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.horizon_hours = horizon_hours
        self.same_candle_strategy = "heuristic"  # "conservative" | "heuristic" | "favor_tp"

    def load_feature_data(self) -> pd.DataFrame:
        """Load engineered features data."""
        logger.info("Loading engineered features data...")

        # Try datasets directory first (new structure), then root (compatibility)
        datasets_dir = self.output_dir / 'datasets'
        features_file = datasets_dir / 'features_engineered.parquet'

        if not features_file.exists():
            # Fallback to root directory for backward compatibility
            features_file = self.output_dir / 'features_engineered.parquet'

        if not features_file.exists():
            logger.error(f"Features file not found: {features_file}")
            sys.exit(1)

        try:
            df = pd.read_parquet(features_file)
            logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            sys.exit(1)

    def create_percentage_based_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create labels based on percentage TP/SL.
        
        Label:
          +1: LONG TP hit before SL (profitable long)
          -1: SHORT TP hit before SL (profitable short)
           0: No clear edge / both hit in same candle / neither hit
        """
        logger.info("Creating percentage-based labels...")
        logger.info(f"TP: {self.tp_pct*100:.1f}%, SL: {self.sl_pct*100:.1f}%, Horizon: {self.horizon_hours} hours")

        df = df.copy()
        # Handle both raw column names and price_* prefixed names
        close_col = 'price_close' if 'price_close' in df.columns else 'close'
        high_col = 'price_high' if 'price_high' in df.columns else 'high'
        low_col = 'price_low' if 'price_low' in df.columns else 'low'
        open_col = 'price_open' if 'price_open' in df.columns else 'open'
        
        close = df[close_col].values
        high = df[high_col].values
        low = df[low_col].values
        open_ = df[open_col].values

        n = len(df)
        labels_long = np.zeros(n, dtype=np.int64)  # Labels untuk LONG model
        labels_short = np.zeros(n, dtype=np.int64)   # Labels untuk SHORT model

        for i in range(n):
            if i + 1 >= n:
                break

            entry_price = close[i]

            # Calculate TP/SL levels untuk LONG
            tp_long = entry_price * (1.0 + self.tp_pct)
            sl_long = entry_price * (1.0 - self.sl_pct)

            # Calculate TP/SL levels untuk SHORT
            tp_short = entry_price * (1.0 - self.tp_pct)
            sl_short = entry_price * (1.0 + self.sl_pct)

            # Look ahead
            end = min(n, i + self.horizon_hours + 1)
            future_high = high[i+1:end]
            future_low = low[i+1:end]
            future_open = open_[i+1:end] if i+1 < len(open_) else [close[i]]
            future_close = close[i+1:end] if i+1 < len(close) else [close[i]]

            # Track which event happens first
            long_hit_tp_idx = None
            long_hit_sl_idx = None
            short_hit_tp_idx = None
            short_hit_sl_idx = None

            for j in range(len(future_high)):
                h = future_high[j]
                l = future_low[j]

                # LONG: Check TP and SL
                tp_triggered = (h >= tp_long)
                sl_triggered = (l <= sl_long)

                if tp_triggered and sl_triggered:
                    # Both triggered in same candle
                    if self.same_candle_strategy == "conservative":
                        if long_hit_tp_idx is None:
                            long_hit_tp_idx = j
                        if long_hit_sl_idx is None:
                            long_hit_sl_idx = j
                    elif self.same_candle_strategy == "favor_tp":
                        if long_hit_tp_idx is None:
                            long_hit_tp_idx = j
                    elif self.same_candle_strategy == "heuristic":
                        o = future_open[j] if j < len(future_open) else close[i]
                        c = future_close[j] if j < len(future_close) else close[i]
                        mid = (h + l) / 2
                        if c > mid:  # Bullish candle → favor TP
                            if long_hit_tp_idx is None:
                                long_hit_tp_idx = j
                        else:  # Bearish candle → favor SL
                            if long_hit_sl_idx is None:
                                long_hit_sl_idx = j
                else:
                    if long_hit_tp_idx is None and tp_triggered:
                        long_hit_tp_idx = j
                    if long_hit_sl_idx is None and sl_triggered:
                        long_hit_sl_idx = j

                # SHORT: Check TP and SL
                tp_triggered_short = (l <= tp_short)
                sl_triggered_short = (h >= sl_short)

                if tp_triggered_short and sl_triggered_short:
                    if self.same_candle_strategy == "conservative":
                        if short_hit_tp_idx is None:
                            short_hit_tp_idx = j
                        if short_hit_sl_idx is None:
                            short_hit_sl_idx = j
                    elif self.same_candle_strategy == "favor_tp":
                        if short_hit_tp_idx is None:
                            short_hit_tp_idx = j
                    elif self.same_candle_strategy == "heuristic":
                        o = future_open[j] if j < len(future_open) else close[i]
                        c = future_close[j] if j < len(future_close) else close[i]
                        mid = (h + l) / 2
                        if c < mid:  # Bearish candle → favor SHORT TP
                            if short_hit_tp_idx is None:
                                short_hit_tp_idx = j
                        else:  # Bullish candle → favor SHORT SL
                            if short_hit_sl_idx is None:
                                short_hit_sl_idx = j
                else:
                    if short_hit_tp_idx is None and tp_triggered_short:
                        short_hit_tp_idx = j
                    if short_hit_sl_idx is None and sl_triggered_short:
                        short_hit_sl_idx = j

            # Determine LONG label
            long_edge = False
            if long_hit_tp_idx is not None:
                if long_hit_sl_idx is None:
                    long_edge = True
                elif long_hit_tp_idx < long_hit_sl_idx:
                    long_edge = True

            # Determine SHORT label
            short_edge = False
            if short_hit_tp_idx is not None:
                if short_hit_sl_idx is None:
                    short_edge = True
                elif short_hit_tp_idx < short_hit_sl_idx:
                    short_edge = True

            # Assign labels
            labels_long[i] = 1 if long_edge else 0
            labels_short[i] = 1 if short_edge else 0

        df[self.label_col_long] = labels_long
        df[self.label_col_short] = labels_short

        # Drop last few bars yang tidak bisa di-label
        df = df.iloc[:-self.horizon_hours].reset_index(drop=True)

        # Log label distribution
        logger.info("Label Distribution:")
        logger.info(f"LONG - Profitable (1): {df[self.label_col_long].sum()} ({df[self.label_col_long].mean()*100:.1f}%)")
        logger.info(f"LONG - Not Profitable (0): {(df[self.label_col_long] == 0).sum()} ({(df[self.label_col_long] == 0).mean()*100:.1f}%)")
        logger.info(f"SHORT - Profitable (1): {df[self.label_col_short].sum()} ({df[self.label_col_short].mean()*100:.1f}%)")
        logger.info(f"SHORT - Not Profitable (0): {(df[self.label_col_short] == 0).sum()} ({(df[self.label_col_short] == 0).mean()*100:.1f}%)")

        return df

    def validate_labels(self, df: pd.DataFrame) -> bool:
        """Validate label quality and distribution."""
        logger.info("\n=== Label Validation ===")

        if self.label_col_long not in df.columns or self.label_col_short not in df.columns:
            logger.error("Label columns not found")
            return False

        # Basic statistics
        total_samples = len(df)
        logger.info(f"Total samples: {total_samples}")

        # Check for reasonable balance
        long_pct = df[self.label_col_long].mean() * 100
        short_pct = df[self.label_col_short].mean() * 100

        if long_pct < 5 or long_pct > 50:
            logger.warning(f"LONG label distribution may be imbalanced: {long_pct:.1f}% profitable")
        else:
            logger.info(f"LONG label distribution: {long_pct:.1f}% profitable")

        if short_pct < 5 or short_pct > 50:
            logger.warning(f"SHORT label distribution may be imbalanced: {short_pct:.1f}% profitable")
        else:
            logger.info(f"SHORT label distribution: {short_pct:.1f}% profitable")

        logger.info("=" * 30)
        return True

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features and labels for training."""
        logger.info("Preparing training data...")

        # Get feature columns (exclude metadata and labels)
        exclude_cols = ['timestamp', 'time', self.label_col_long, self.label_col_short]
        # Also exclude raw OHLCV if we have price_* versions
        exclude_cols.extend(['open', 'high', 'low', 'close', 'volume'])

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Ensure we have features
        if not feature_cols:
            logger.error("No feature columns found!")
            sys.exit(1)

        logger.info(f"Using {len(feature_cols)} features for training")

        # Prepare X and y
        X = df[feature_cols].copy()
        y_long = df[self.label_col_long].copy()
        y_short = df[self.label_col_short].copy()

        # Handle any remaining NaN values in features
        X = X.fillna(0)
        y_long = y_long[X.index]
        y_short = y_short[X.index]

        # Remove any constant features
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            logger.warning(f"Removing {len(constant_features)} constant features: {constant_features}")
            X = X.drop(columns=constant_features)

        logger.info(f"Final training set: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y_long, y_short

    def save_labeled_data(self, df: pd.DataFrame, X: pd.DataFrame, y_long: pd.Series, y_short: pd.Series):
        """Save labeled dataset and training components."""
        if df.empty:
            logger.error("No data to save")
            return

        # Create datasets directory
        datasets_dir = self.output_dir / 'datasets'
        datasets_dir.mkdir(exist_ok=True)

        # Save complete labeled dataset
        labeled_file = datasets_dir / 'labeled_data.parquet'
        df.to_parquet(labeled_file, index=False)
        logger.info(f"Labeled dataset saved to {labeled_file}")

        # Save training features and labels separately
        X_file = datasets_dir / 'X_train_features.parquet'
        X.to_parquet(X_file, index=False)
        logger.info(f"Features saved to {X_file}")

        y_long_file = datasets_dir / 'y_train_labels_long.parquet'
        y_long.to_frame().to_parquet(y_long_file, index=False)
        logger.info(f"LONG labels saved to {y_long_file}")

        y_short_file = datasets_dir / 'y_train_labels_short.parquet'
        y_short.to_frame().to_parquet(y_short_file, index=False)
        logger.info(f"SHORT labels saved to {y_short_file}")

        # Save time index aligned with X/y for split reproducibility
        time_file = datasets_dir / 'time_index.parquet'
        try:
            if 'timestamp' in df.columns:
                time_index = df.loc[X.index, ['timestamp']].copy()
                time_index.to_parquet(time_file, index=False)
                logger.info(f"Time index saved to {time_file}")
            elif 'time' in df.columns:
                time_index = df.loc[X.index, ['time']].copy()
                time_index.to_parquet(time_file, index=False)
                logger.info(f"Time index saved to {time_file}")
        except Exception as e:
            logger.warning(f"Failed to save time index: {e}")

        # Save feature list for training
        feature_list_file = self.output_dir / 'training_features.txt'
        with open(feature_list_file, 'w') as f:
            f.write("Training Features:\n")
            f.write("=" * 20 + "\n")
            for feature in X.columns:
                f.write(f"{feature}\n")
        logger.info(f"Training feature list saved to {feature_list_file}")

        # Save dataset summary
        import pytz
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        current_time = datetime.now(jakarta_tz)

        dataset_dir = self.output_dir / 'datasets' / 'summary'
        dataset_dir.mkdir(parents=True, exist_ok=True)

        timestamp = current_time.strftime('%Y%m%d_%H%M%S')
        summary_file = dataset_dir / f'dataset_summary_{timestamp}.txt'

        summary_content = f"""Dataset Summary
{"=" * 20}

Training Session Info:
- Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S WIB')}
- Exchange: {self.data_filter.exchange}
- Pair: {self.data_filter.pair}
- Interval: {self.data_filter.interval}
- TP: {self.tp_pct*100:.1f}%
- SL: {self.sl_pct*100:.1f}%
- Horizon: {self.horizon_hours} hours

Dataset Statistics:
- Total samples: {len(df):,}
- Features: {len(X.columns)}
- LONG label distribution:
  * Profitable (1): {y_long.sum():,} ({y_long.mean()*100:.1f}%)
  * Not Profitable (0): {len(y_long) - y_long.sum():,} ({(1-y_long.mean())*100:.1f}%)
- SHORT label distribution:
  * Profitable (1): {y_short.sum():,} ({y_short.mean()*100:.1f}%)
  * Not Profitable (0): {len(y_short) - y_short.sum():,} ({(1-y_short.mean())*100:.1f}%)

Time Range:
- Start: {df['timestamp'].min() if 'timestamp' in df.columns else 'N/A'}
- End: {df['timestamp'].max() if 'timestamp' in df.columns else 'N/A'}

Feature Columns:
{chr(10).join(f"- {feature}" for feature in sorted(X.columns))}
"""

        with open(summary_file, 'w') as f:
            f.write(summary_content)

        logger.info(f"Dataset summary saved to {summary_file}")

def main():
    """Main function to build labels."""
    args = parse_arguments()
    validate_arguments(args)

    # Set up logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create data filter
    data_filter = DataFilter(args)

    # Initialize label builder with args
    builder = LabelBuilder(
        data_filter, 
        args.output_dir,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        horizon_hours=args.horizon_hours
    )

    try:
        # Load feature data
        logger.info("Loading feature data for label building...")
        df = builder.load_feature_data()

        # Create labels
        df = builder.create_percentage_based_labels(df)

        # Validate labels
        if not builder.validate_labels(df):
            sys.exit(1)

        # Prepare training data
        X, y_long, y_short = builder.prepare_training_data(df)

        # Save labeled data
        builder.save_labeled_data(df, X, y_long, y_short)

        logger.info("\n=== Label Building Complete ===")
        logger.info("Ready for xgboost_trainer.py")

    except Exception as e:
        logger.error(f"Error in label building: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
