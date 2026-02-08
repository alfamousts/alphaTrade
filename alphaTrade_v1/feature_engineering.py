#!/usr/bin/env python3
"""
Feature engineering for AlphaTrade pipeline.
Creates OHLCV-based features compatible with QuantConnect deployment.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
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

class FeatureEngineer:
    """Implement feature engineering based on OHLCV data."""

    def __init__(self, data_filter: DataFilter, output_dir: str = './output_train'):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.feature_columns = []

    def load_price_data(self) -> pd.DataFrame:
        """Load price data from previous step."""
        logger.info("Loading price data...")

        # Try datasets directory first (new structure), then root (compatibility)
        datasets_dir = self.output_dir / 'datasets'
        raw_data_dir = datasets_dir / 'raw'
        price_file = raw_data_dir / 'hourly_price.parquet'

        if not price_file.exists():
            # Fallback to root directory for backward compatibility
            price_file = self.output_dir / 'hourly_price.parquet'

        if not price_file.exists():
            logger.error(f"Price data file not found: {price_file}")
            logger.error(f"Checked: {raw_data_dir / 'hourly_price.parquet'} and {self.output_dir / 'hourly_price.parquet'}")
            sys.exit(1)

        try:
            df = pd.read_parquet(price_file)
            logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading price data: {e}")
            sys.exit(1)

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for OHLCV data (matching QuantConnect implementation)."""
        logger.info("Adding price features...")

        # ===== Base Price Features =====
        df['price_open'] = df['open']
        df['price_high'] = df['high']
        df['price_low'] = df['low']
        df['price_close'] = df['close']
        df['price_volume_usd'] = df['close'] * df['volume']

        # ===== Return Features =====
        df['price_close_return_1'] = df['close'].pct_change(1)
        df['price_close_return_5'] = df['close'].pct_change(5)
        df['price_log_return'] = np.log(df['close'] / df['close'].shift(1))

        # ===== Volatility Features =====
        returns = df['close'].pct_change()
        df['price_rolling_vol_5'] = returns.rolling(window=5).std()
        df['price_close_mean_5'] = df['close'].rolling(window=5).mean()
        df['price_close_std_5'] = df['close'].rolling(window=5).std()
        df['price_true_range'] = df['high'] - df['low']

        # ===== Volume Features =====
        df['price_volume_mean_10'] = df['volume'].rolling(window=10).mean()
        volume_mean = df['volume'].rolling(window=50).mean()
        volume_std = df['volume'].rolling(window=50).std()
        df['price_volume_zscore'] = (df['volume'] - volume_mean) / (volume_std + 1e-8)
        df['price_volume_change'] = df['volume'].pct_change(1)

        # ===== Candlestick Features =====
        df['price_wick_upper'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['price_wick_lower'] = df[['open', 'close']].min(axis=1) - df['low']
        df['price_body_size'] = np.abs(df['close'] - df['open'])

        # ===== Additional Technical Indicators =====
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_lower'] = sma20 - (std20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (sma20 + 1e-8)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

        # Momentum
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1

        # Price position in range
        high_20 = df['high'].rolling(window=20).max()
        low_20 = df['low'].rolling(window=20).min()
        df['price_position'] = (df['close'] - low_20) / (high_20 - low_20 + 1e-8)

        # Trend strength
        df['trend_strength'] = abs(df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-8)

        # Define feature columns (matching QuantConnect)
        self.feature_columns = [
            'price_open', 'price_high', 'price_low', 'price_close', 'price_volume_usd',
            'price_close_return_1', 'price_close_return_5', 'price_log_return',
            'price_rolling_vol_5', 'price_true_range', 'price_close_mean_5',
            'price_close_std_5', 'price_volume_mean_10', 'price_volume_zscore',
            'price_volume_change', 'price_wick_upper', 'price_wick_lower',
            'price_body_size', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position', 'momentum_10', 'price_position', 'trend_strength'
        ]

        return df

    def remove_stationary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove non-stationary features using ADF test."""
        logger.info("Checking feature stationarity...")
        
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            logger.warning("statsmodels not available, skipping stationarity check")
            return df

        def is_stationary(series):
            """Check if series is stationary using ADF test."""
            try:
                result = adfuller(series.dropna())
                return result[1] < 0.05  # p-value < 0.05 means stationary
            except:
                return False

        feature_cols = [col for col in self.feature_columns if col in df.columns]
        dropped_features = []
        
        for col in feature_cols:
            if not is_stationary(df[col]):
                logger.info(f"Dropping non-stationary feature: {col}")
                dropped_features.append(col)
                if col in self.feature_columns:
                    self.feature_columns.remove(col)

        if dropped_features:
            logger.info(f"Dropped {len(dropped_features)} non-stationary features")
        else:
            logger.info("All features are stationary")

        return df

    def remove_correlated_features(self, df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
        """Remove highly correlated features."""
        logger.info(f"Removing highly correlated features (threshold={threshold})...")

        feature_cols = [col for col in self.feature_columns if col in df.columns]
        if len(feature_cols) < 2:
            return df

        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr().abs()

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > threshold:
                    feat1 = corr_matrix.columns[i]
                    feat2 = corr_matrix.columns[j]
                    high_corr_pairs.append((feat1, feat2, corr_value))

        # Drop features (keep the one with lower index)
        features_to_drop = set()
        for feat1, feat2, corr_val in high_corr_pairs:
            feat1_idx = feature_cols.index(feat1) if feat1 in feature_cols else len(feature_cols)
            feat2_idx = feature_cols.index(feat2) if feat2 in feature_cols else len(feature_cols)
            
            if feat2_idx > feat1_idx:
                features_to_drop.add(feat2)
            else:
                features_to_drop.add(feat1)

        if features_to_drop:
            logger.info(f"Dropping {len(features_to_drop)} highly correlated features: {sorted(features_to_drop)}")
            for feat in features_to_drop:
                if feat in self.feature_columns:
                    self.feature_columns.remove(feat)

        return df

    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features for ML."""
        logger.info("Cleaning features...")

        # Replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Get feature columns only
        feature_cols = [col for col in self.feature_columns if col in df.columns]

        # Fill missing values with forward fill then backward fill
        for col in feature_cols:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].ffill().bfill()

        # Drop rows with missing essential features
        essential_features = ['price_close_return_1', 'price_rolling_vol_5']
        available_essential = [col for col in essential_features if col in df.columns]

        if available_essential:
            initial_rows = len(df)
            df = df.dropna(subset=available_essential)
            final_rows = len(df)
            logger.info(f"Dropped {initial_rows - final_rows} rows with missing essential features")

        # Report feature statistics
        feature_df = df[feature_cols]
        missing_counts = feature_df.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values in features after cleaning:\n{missing_counts[missing_counts > 0]}")

        logger.info(f"Final feature set: {len(feature_cols)} features")
        return df

    def validate_features(self, df: pd.DataFrame) -> bool:
        """Validate engineered features."""
        logger.info("\n=== Feature Validation ===")

        feature_cols = [col for col in self.feature_columns if col in df.columns]
        logger.info(f"Total engineered features: {len(feature_cols)}")

        if not feature_cols:
            logger.error("No features were created!")
            return False

        # Check for basic statistics
        feature_stats = df[feature_cols].describe()
        logger.info("Feature statistics summary:")
        logger.info(feature_stats.loc[['mean', 'std', 'min', 'max']].round(4))

        # Check for constant features
        constant_features = []
        for col in feature_cols:
            if df[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            logger.warning(f"Constant features found: {constant_features}")
        else:
            logger.info("No constant features found")

        logger.info("=" * 30)
        return True

    def save_features(self, df: pd.DataFrame):
        """Save engineered features."""
        if df.empty:
            logger.error("No data to save")
            return

        # Create directories
        datasets_dir = self.output_dir / 'datasets'
        features_dir = self.output_dir / 'features'
        datasets_dir.mkdir(exist_ok=True)
        features_dir.mkdir(exist_ok=True)

        # Save complete dataset with features to datasets directory
        output_file = datasets_dir / 'features_engineered.parquet'
        df.to_parquet(output_file, index=False)
        logger.info(f"Features saved to {output_file}")

        # Save feature list to features directory
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        feature_file = features_dir / 'feature_list.txt'
        with open(feature_file, 'w') as f:
            f.write("Engineered Features:\n")
            f.write("=" * 20 + "\n")
            for col in feature_cols:
                f.write(f"{col}\n")
        logger.info(f"Feature list saved to {feature_file}")

        # Save features-only dataset to datasets directory
        features_df = df[feature_cols].copy()
        features_file = datasets_dir / 'features_only.parquet'
        features_df.to_parquet(features_file, index=False)
        logger.info(f"Features-only dataset saved to {features_file}")

        # Also keep copies in root for backward compatibility
        root_output_file = self.output_dir / 'features_engineered.parquet'
        root_features_file = self.output_dir / 'features_only.parquet'
        root_feature_file = self.output_dir / 'feature_list.txt'

        try:
            df.to_parquet(root_output_file, index=False)
            features_df.to_parquet(root_features_file, index=False)
            root_feature_file.write_text(feature_file.read_text())
            logger.info("Also saved copies to root directory for compatibility")
        except Exception as e:
            logger.warning(f"Could not save root compatibility copies: {e}")

def main():
    """Main function to engineer features."""
    args = parse_arguments()
    validate_arguments(args)

    # Set up logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create data filter
    data_filter = DataFilter(args)

    # Initialize feature engineer
    engineer = FeatureEngineer(data_filter, args.output_dir)

    try:
        # Load price data
        logger.info("Loading price data for feature engineering...")
        df = engineer.load_price_data()

        # Add features
        df = engineer.add_price_features(df)

        # Remove non-stationary features (optional)
        # df = engineer.remove_stationary_features(df)

        # Remove highly correlated features (optional)
        # df = engineer.remove_correlated_features(df, threshold=0.9)

        # Clean features
        df = engineer.clean_features(df)

        # Validate features
        if not engineer.validate_features(df):
            sys.exit(1)

        # Save features
        engineer.save_features(df)

        logger.info("\n=== Feature Engineering Complete ===")
        logger.info("Ready for label_builder.py")

    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
