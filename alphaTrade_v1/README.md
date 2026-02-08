# AlphaTrade XGBoost Training Pipeline

Modular pipeline for training XGBoost models for crypto trading using percentage-based TP/SL strategy.

## Overview

This pipeline trains separate XGBoost models for LONG and SHORT predictions based on percentage-based take profit (TP) and stop loss (SL) labels. The pipeline follows the same structure as `v1_futures_new_gen` but adapted for SQLite database and percentage-based labeling.

## Pipeline Steps

1. **Load Database** (`load_database.py`)
   - Loads hourly price data from SQLite database
   - Filters by exchange, pair, and time range
   - Saves raw data to parquet

2. **Feature Engineering** (`feature_engineering.py`)
   - Creates OHLCV-based features (matching QuantConnect implementation)
   - Technical indicators: RSI, MACD, Bollinger Bands, Momentum
   - Removes non-stationary and highly correlated features
   - Outputs engineered features

3. **Label Building** (`label_builder.py`)
   - Creates percentage-based TP/SL labels
   - Separate labels for LONG and SHORT models
   - Labels: 1 = profitable (TP hit before SL), 0 = not profitable
   - Horizon: configurable (default: 24 hours = 1 day)

4. **Model Training** (`xgboost_trainer.py`)
   - Trains separate XGBoost models for LONG and SHORT
   - Time-based train/test split (80/20)
   - Handles class imbalance with `scale_pos_weight`
   - Cross-validation and feature importance analysis
   - Saves models and results

5. **Model Upload** (`model_uploader.py`) - Optional
   - Uploads trained models to API server
   - Base64 encodes models for transmission
   - Supports automatic versioning

## Usage

### Basic Usage

```bash
cd alphatrade
./simple_run.sh
```

### With Custom Parameters

```bash
EXCHANGE=binance \
PAIR=btcusdt.p \
TP_PCT=0.02 \
SL_PCT=0.01 \
HORIZON_HOURS=24 \
./simple_run.sh
```

### With Time Filter

```bash
./simple_run.sh --days 365
```

### With Model Upload

```bash
./simple_run.sh --upload-to-api --model-version alphatrade_btc_v1
```

### Individual Steps

```bash
# Step 1: Load data
python load_database.py --exchange binance --pair btcusdt.p --db-path ../sqlite_crypto.db

# Step 2: Feature engineering
python feature_engineering.py --exchange binance --pair btcusdt.p

# Step 3: Label building
python label_builder.py --exchange binance --pair btcusdt.p --tp-pct 0.02 --sl-pct 0.01 --horizon-hours 24

# Step 4: Model training
python xgboost_trainer.py --exchange binance --pair btcusdt.p

# Step 5: Upload models (optional)
python model_uploader.py --exchange binance --pair btcusdt.p --upload-to-api --model-version alphatrade_btc_v1
```

## Configuration Parameters

### Data Loading
- `--exchange`: Exchange name (default: binance)
- `--pair`: Trading pair (default: btcusdt.p)
- `--db-path`: Path to SQLite database (default: ../sqlite_crypto.db)
- `--days`: Number of recent days to include
- `--time`: Time range as Unix timestamps (ms): start_time,end_time

### Label Configuration
- `--tp-pct`: Take profit percentage (default: 0.02 = 2%)
- `--sl-pct`: Stop loss percentage (default: 0.01 = 1%)
- `--horizon-hours`: Label horizon in hours (default: 24 = 1 day)

### Model Configuration
- `--skip-tuning`: Skip hyperparameter tuning
- `--verbose`: Enable verbose logging

### API Upload
- `--upload-to-api`: Enable model upload to API
- `--model-version`: Model version string (required for upload)
- `--api-domain`: API domain (default: https://api.dragonfortune.ai)
- `--api-version`: API version (default: v1)

## Output Structure

```
output_train_alphatrade/
├── models/
│   ├── model_long_TIMESTAMP.joblib
│   ├── model_short_TIMESTAMP.joblib
│   ├── latest_model_long.joblib
│   └── latest_model_short.joblib
├── datasets/
│   ├── raw/
│   │   └── hourly_price.parquet
│   ├── features_engineered.parquet
│   ├── features_only.parquet
│   ├── labeled_data.parquet
│   ├── X_train_features.parquet
│   ├── y_train_labels_long.parquet
│   ├── y_train_labels_short.parquet
│   └── summary/
│       └── dataset_summary_TIMESTAMP.txt
├── features/
│   └── feature_list.txt
├── training_results_TIMESTAMP.json
├── feature_importance_long_TIMESTAMP.csv
├── feature_importance_short_TIMESTAMP.csv
└── model_features.txt
```

## Features

The pipeline creates the following features (matching QuantConnect implementation):

### Base Price Features
- `price_open`, `price_high`, `price_low`, `price_close`, `price_volume_usd`

### Return Features
- `price_close_return_1`, `price_close_return_5`, `price_log_return`

### Volatility Features
- `price_rolling_vol_5`, `price_close_mean_5`, `price_close_std_5`, `price_true_range`

### Volume Features
- `price_volume_mean_10`, `price_volume_zscore`, `price_volume_change`

### Candlestick Features
- `price_wick_upper`, `price_wick_lower`, `price_body_size`

### Technical Indicators
- `rsi`, `macd`, `macd_signal`, `macd_hist`
- `bb_width`, `bb_position`
- `momentum_10`
- `price_position`, `trend_strength`

## Model Training

### XGBoost Parameters (Default)
```python
{
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 500,
    'early_stopping_rounds': 10,
    'scale_pos_weight': <calculated from data>
}
```

### Data Splitting
- **Method**: Time-based split (maintains temporal order)
- **Ratio**: 80% training, 20% testing
- **Exclusion**: Year 2025 excluded from training (reserved for backtesting)

## Label Strategy

### Percentage-Based Labels
- **LONG Model**: 
  - Label = 1 if TP (entry * (1 + tp_pct)) hit before SL (entry * (1 - sl_pct))
  - Label = 0 otherwise

- **SHORT Model**:
  - Label = 1 if TP (entry * (1 - tp_pct)) hit before SL (entry * (1 + sl_pct))
  - Label = 0 otherwise

### Same Candle Strategy
When TP and SL are both triggered in the same candle:
- **heuristic**: Uses candle close position to infer direction
- **conservative**: Labels as 0 (no edge)
- **favor_tp**: Assumes TP wins

## API Integration

Models can be uploaded to the API server for QuantConnect integration:

```bash
./simple_run.sh --upload-to-api --model-version alphatrade_btc_v1
```

The API endpoint expects:
- **POST** `/api/v1/alphatrade/latest/model`
- **Payload**: JSON with `model_version`, `model_data_base64`, `feature_names`, `created_at`

QuantConnect will automatically download models when the version changes.

## Differences from v1_futures_new_gen

1. **Data Source**: SQLite database instead of MySQL
2. **Features**: Only OHLCV features (no funding, basis, OI, L/S ratios)
3. **Labels**: Percentage-based TP/SL instead of simple up/down
4. **Models**: Separate LONG and SHORT models instead of single binary classifier
5. **Database**: Single table (`hourly_price`) instead of 6 tables

## Requirements

Install all dependencies from `requirements.txt`:

```bash
cd alphatrade
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy pyarrow xgboost scikit-learn joblib requests statsmodels pytz pymysql SQLAlchemy python-dotenv
```

**Note**: `pyarrow` is required for parquet file support. If you see errors about missing parquet engine, install it:
```bash
pip install pyarrow
```

## Notes

- Models are trained with only OHLCV features to match QuantConnect deployment
- Separate models for LONG and SHORT allow for different trading strategies
- Time-based split ensures no data leakage
- Class weights automatically calculated to handle imbalanced labels

## Label Strategy Rationale

The default parameters (2% TP, 1% SL, 24h horizon) are optimized for:
- **More training samples**: 24h horizon provides ~7x more samples than 168h
- **Realistic targets**: 2% TP is more achievable in crypto markets than 7.5%
- **Better risk/reward**: 2:1 ratio balances profitability with sample size
- **Reduced regime risk**: 1-day horizon is more predictable than 7 days
- **Profitability focus**: Still maintains TP/SL logic (better than simple direction prediction)

You can adjust these parameters based on your trading strategy:
- **Scalping**: Lower TP/SL (0.5-1%), shorter horizon (1-6 hours)
- **Day trading**: Current defaults (2%/1%, 24h)
- **Swing trading**: Higher TP/SL (3-5%), longer horizon (48-72h)
