# AlphaTrade

A modular machine learning pipeline for training XGBoost models for cryptocurrency futures trading. This repository provides a complete end-to-end solution for data loading, feature engineering, label building, model training, and deployment.

## 📋 Overview

AlphaTrade is designed to train binary classification models that predict profitable trading opportunities based on percentage-based take profit (TP) and stop loss (SL) strategies. The pipeline is optimized for futures trading on major cryptocurrency exchanges (Binance, Bybit) and integrates seamlessly with QuantConnect for live trading.

### Key Features

- **Modular Architecture**: Each pipeline step is a separate Python script for easy maintenance and debugging
- **Version Isolation**: Each version (v1, v2, etc.) has its own isolated codebase and outputs
- **Dual Model Training**: Separate XGBoost models for LONG and SHORT predictions
- **Percentage-Based Labeling**: Labels based on realistic TP/SL targets (e.g., 2% TP, 1% SL)
- **MySQL Integration**: Loads data from remote MySQL database (`cg_futures_price_history`)
- **Hyperparameter Tuning**: Automated grid search for optimal model parameters
- **Threshold Calibration**: Finds optimal prediction thresholds for precision/F1 score
- **Production Ready**: Models can be uploaded to API server for QuantConnect integration

## 🏗️ Repository Structure

```
alphatrade/
├── README.md                    # This file - repository overview
├── .gitignore                   # Git ignore rules (excludes output_train)
└── alphaTrade_v1/               # Version 1 implementation
    ├── README.md                # Version-specific documentation
    ├── requirements.txt         # Python dependencies
    ├── simple_run.sh            # Main pipeline orchestrator
    ├── command_line_options.py  # CLI argument parsing
    ├── load_database.py         # Data loading from MySQL
    ├── feature_engineering.py   # Feature creation
    ├── label_builder.py         # TP/SL label generation
    ├── xgboost_trainer.py       # Model training
    ├── model_uploader.py        # Model deployment
    └── output_train/            # Training outputs (git-ignored)
        ├── datasets/            # Processed data files
        ├── models/              # Trained model files
        └── ...
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with pip
2. **MySQL Database Access** (configured in `.env` file)
3. **Required Environment Variables**:
   ```bash
   TRADING_DB_HOST=your_host
   TRADING_DB_PORT=3306
   TRADING_DB_USER=your_user
   TRADING_DB_PASSWORD=your_password
   TRADING_DB_NAME=newera
   ```

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/alfamousts/alphaTrade.git
   cd alphaTrade
   ```

2. **Navigate to version directory**:
   ```bash
   cd alphaTrade_v1
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   # Create .env file in alphaTrade_v1/ directory
   cp .env.example .env  # If you have an example file
   # Or create .env manually with database credentials
   ```

5. **Run the pipeline**:
   ```bash
   ./simple_run.sh
   ```

For detailed usage instructions, see [alphaTrade_v1/README.md](alphaTrade_v1/README.md).

## 📊 Pipeline Workflow

The training pipeline consists of 5 main steps:

1. **Data Loading** → Load OHLCV data from MySQL database
2. **Feature Engineering** → Create technical indicators and derived features
3. **Label Building** → Generate TP/SL-based binary labels
4. **Model Training** → Train separate LONG/SHORT XGBoost models
5. **Model Upload** (Optional) → Deploy models to API server

### Data Flow

```
MySQL Database
    ↓
[load_database.py] → Raw OHLCV data (Parquet)
    ↓
[feature_engineering.py] → Engineered features (Parquet)
    ↓
[label_builder.py] → Labeled dataset (Parquet)
    ↓
[xgboost_trainer.py] → Trained models (.joblib)
    ↓
[model_uploader.py] → API Server (Base64 encoded)
```

## 🎯 Versioning Strategy

This repository uses **version-based isolation**:

- Each version (`alphaTrade_v1`, `alphaTrade_v2`, etc.) is completely independent
- Each version has its own:
  - Source code
  - Output directory (`output_train/`)
  - Configuration parameters
  - Documentation

This allows you to:
- Compare different model architectures
- Test new features without breaking existing versions
- Maintain multiple production models simultaneously
- Easily roll back to previous versions

## 📈 Model Architecture

### Label Strategy

- **Default Parameters**:
  - Take Profit: 2% (`tp_pct=0.02`)
  - Stop Loss: 1% (`sl_pct=0.01`)
  - Horizon: 24 hours (`horizon_hours=24`)

- **Label Logic**:
  - `label=1` (LONG): Price hits TP before SL within horizon
  - `label=0` (LONG): Price hits SL before TP, or neither within horizon
  - Separate labels for SHORT model (inverted logic)

### Model Training

- **Algorithm**: XGBoost (Gradient Boosting)
- **Models**: Separate models for LONG and SHORT predictions
- **Data Split**: Time-based (train/validation/test)
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Threshold Calibration**: Optimizes precision/F1 thresholds

## 🔧 Configuration

### Command Line Arguments

Key parameters can be customized via environment variables or CLI flags:

```bash
# Example: Custom TP/SL and horizon
TP_PCT=0.03 SL_PCT=0.015 HORIZON_HOURS=48 ./simple_run.sh

# Example: Different exchange/pair
EXCHANGE=Bybit PAIR=ETHUSDT ./simple_run.sh

# Example: Skip hyperparameter tuning
./simple_run.sh --skip-tuning
```

See `alphaTrade_v1/README.md` for complete parameter documentation.

## 📦 Output Files

After running the pipeline, `output_train/` contains:

- **datasets/**: Processed data files (Parquet format)
  - `raw/`: Raw OHLCV data
  - `features_engineered.parquet`: Engineered features
  - `labeled_data.parquet`: Features + labels
  - `X_train_features.parquet`: Training features
  - `y_train_labels_long.parquet`: LONG labels
  - `y_train_labels_short.parquet`: SHORT labels
  - `summary/`: Dataset summaries

- **models/**: Trained model files
  - `model_long_*.joblib`: LONG prediction model
  - `model_short_*.joblib`: SHORT prediction model
  - `latest_model_long.joblib`: Latest LONG model
  - `latest_model_short.joblib`: Latest SHORT model

- **Results**: Training metrics, feature importance, thresholds
  - `training_results_*.json`: Training metrics
  - `model_thresholds.json`: Optimal thresholds for QuantConnect
  - `feature_importance_*.csv`: Feature importance rankings

## 🔗 Integration

### QuantConnect Integration

Trained models are designed to work with QuantConnect algorithms:

1. Models are uploaded to API server via `model_uploader.py`
2. QuantConnect script (`qc_futures_new_gen_v6.py`) loads models via API
3. Models are refreshed automatically when new versions are available
4. Only OHLCV features are used in live trading (matching training)

### API Server

The pipeline supports uploading models to a remote API server:

- **Endpoint**: Configurable via `model_uploader.py`
- **Format**: Base64-encoded joblib files
- **Versioning**: Automatic version tracking
- **Metadata**: Includes feature names, thresholds, and training info

## 🛠️ Development

### Adding a New Version

1. Create new directory: `alphaTrade_v2/`
2. Copy structure from `alphaTrade_v1/`
3. Modify as needed
4. Each version is completely isolated

### Contributing

1. Create a new version branch
2. Make changes in version-specific directory
3. Test thoroughly
4. Update version-specific README
5. Submit pull request

## 📝 License



## 👥 Authors

Tsaqif Alfatan Nugraha 

## 🙏 Acknowledgments

- Based on `backtest/v1_futures_new_gen` pipeline
- Inspired by `cnn-trade/training_pipeline.ipynb`
- Designed for integration with QuantConnect platform

## 📚 Additional Resources

- [Version 1 Documentation](alphaTrade_v1/README.md) - Detailed v1 pipeline documentation
- [QuantConnect Integration](backtest/v1_futures_new_gen/qc_futures_new_gen_v6.py) - Live trading script
- [Original Pipeline](backtest/v1_futures_new_gen/) - Reference implementation

---

**Note**: The `output_train/` directory is git-ignored to keep the repository clean. Each version maintains its own isolated output directory.
