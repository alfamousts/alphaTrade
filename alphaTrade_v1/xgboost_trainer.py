#!/usr/bin/env python3
"""
XGBoost trainer for binary classification of LONG and SHORT models.
Trains separate models for LONG and SHORT predictions.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import json
import shutil
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

# Default decision threshold for converting probabilities to class labels.
DEFAULT_DECISION_THRESHOLD = 0.43

class XGBoostTrainer:
    """Train XGBoost models for LONG and SHORT predictions."""

    def __init__(self, data_filter: DataFilter, output_dir: str = './output_train'):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.model_long = None
        self.model_short = None
        self.feature_names = None
        self.best_params_long = None
        self.best_params_short = None
        self.time_index = None
        self.split_info = {}
        self.sample_count = 0

    def load_training_data(self) -> tuple:
        """Load prepared training data."""
        logger.info("Loading training data...")

        # Try datasets directory first (new structure), then root (compatibility)
        datasets_dir = self.output_dir / 'datasets'
        X_file = datasets_dir / 'X_train_features.parquet'
        y_long_file = datasets_dir / 'y_train_labels_long.parquet'
        y_short_file = datasets_dir / 'y_train_labels_short.parquet'

        if not X_file.exists() or not y_long_file.exists() or not y_short_file.exists():
            logger.error("Training data files not found. Run label_builder.py first.")
            sys.exit(1)

        try:
            X = pd.read_parquet(X_file)
            y_long = pd.read_parquet(y_long_file).iloc[:, 0]
            y_short = pd.read_parquet(y_short_file).iloc[:, 0]

            # Try to load time index if available
            time_file = datasets_dir / 'time_index.parquet'
            if not time_file.exists():
                time_file = self.output_dir / 'time_index.parquet'
            if time_file.exists():
                try:
                    time_df = pd.read_parquet(time_file)
                    # Handle both 'timestamp' and 'time' column names
                    if 'timestamp' in time_df.columns:
                        self.time_index = time_df['timestamp']
                    elif 'time' in time_df.columns:
                        self.time_index = time_df['time']
                    else:
                        self.time_index = time_df.iloc[:, 0]
                    logger.info(f"Loaded time index: {len(self.time_index)} timestamps")
                except Exception as e:
                    logger.warning(f"Failed to load time index: {e}")
            
            # Fallback: Try to extract timestamps from labeled_data if time_index not available
            if self.time_index is None or len(self.time_index) != len(X):
                try:
                    labeled_file = datasets_dir / 'labeled_data.parquet'
                    if labeled_file.exists():
                        labeled_df = pd.read_parquet(labeled_file)
                        if 'timestamp' in labeled_df.columns:
                            # Align with X index
                            self.time_index = labeled_df.loc[X.index, 'timestamp'].reset_index(drop=True)
                            logger.info(f"Extracted time index from labeled_data: {len(self.time_index)} timestamps")
                        elif 'time' in labeled_df.columns:
                            self.time_index = labeled_df.loc[X.index, 'time'].reset_index(drop=True)
                            logger.info(f"Extracted time index from labeled_data: {len(self.time_index)} timestamps")
                except Exception as e:
                    logger.warning(f"Failed to extract time index from labeled_data: {e}")

            logger.info(f"Loaded features: {X.shape}")
            logger.info(f"Loaded LONG labels: {y_long.shape}")
            logger.info(f"Loaded SHORT labels: {y_short.shape}")
            logger.info(f"LONG label distribution: {y_long.value_counts().to_dict()}")
            logger.info(f"SHORT label distribution: {y_short.value_counts().to_dict()}")

            self.feature_names = list(X.columns)
            self.sample_count = int(len(X))
            return X, y_long, y_short

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            sys.exit(1)

    def prepare_data_splits(self, X: pd.DataFrame, y_long: pd.Series, y_short: pd.Series,
                          test_size: float = 0.2,
                          validation_size: float = 0.2) -> tuple:
        """Prepare train/validation/test splits in time order (no leakage)."""
        logger.info("Preparing data splits (time-based)...")

        n = len(X)
        if n < 100:
            logger.warning("Very small dataset; splits may be unstable")

        test_n = int(n * test_size)
        val_n = int((n - test_n) * validation_size)

        train_end = max(0, n - test_n - val_n)
        val_end = max(train_end, n - test_n)

        X_train = X.iloc[:train_end]
        y_train_long = y_long.iloc[:train_end]
        y_train_short = y_short.iloc[:train_end]
        
        X_val = X.iloc[train_end:val_end]
        y_val_long = y_long.iloc[train_end:val_end]
        y_val_short = y_short.iloc[train_end:val_end]
        
        X_test = X.iloc[val_end:]
        y_test_long = y_long.iloc[val_end:]
        y_test_short = y_short.iloc[val_end:]

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"  LONG: {y_train_long.mean():.3f} profitable")
        logger.info(f"  SHORT: {y_train_short.mean():.3f} profitable")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"  LONG: {y_val_long.mean():.3f} profitable")
        logger.info(f"  SHORT: {y_val_short.mean():.3f} profitable")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"  LONG: {y_test_long.mean():.3f} profitable")
        logger.info(f"  SHORT: {y_test_short.mean():.3f} profitable")
        
        # Display time ranges for verification
        logger.info("")
        logger.info("=" * 70)
        logger.info("TIME-BASED SPLIT VERIFICATION")
        logger.info("=" * 70)

        # Store split info
        self.split_info = {
            "train_end_idx": train_end,
            "val_end_idx": val_end,
            "test_size": test_size,
            "validation_size": validation_size
        }
        
        # Add time ranges if time_index is available
        if self.time_index is not None and len(self.time_index) == len(X):
            try:
                train_time_start = self.time_index.iloc[0] if train_end > 0 else None
                train_time_end = self.time_index.iloc[train_end - 1] if train_end > 0 else None
                val_time_start = self.time_index.iloc[train_end] if val_end > train_end else None
                val_time_end = self.time_index.iloc[val_end - 1] if val_end > train_end else None
                test_time_start = self.time_index.iloc[val_end] if val_end < len(X) else None
                test_time_end = self.time_index.iloc[-1] if val_end < len(X) else None
                
                # Format timestamps for display
                def format_time(ts):
                    if ts is None:
                        return "N/A"
                    if isinstance(ts, pd.Timestamp):
                        return ts.strftime('%Y-%m-%d %H:%M:%S %Z')
                    try:
                        return pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S %Z')
                    except:
                        return str(ts)
                
                train_time_str = f"{format_time(train_time_start)} -> {format_time(train_time_end)}"
                val_time_str = f"{format_time(val_time_start)} -> {format_time(val_time_end)}"
                test_time_str = f"{format_time(test_time_start)} -> {format_time(test_time_end)}"
                
                self.split_info.update({
                    "train_time_range": [str(train_time_start), str(train_time_end)],
                    "val_time_range": [str(val_time_start), str(val_time_end)],
                    "test_time_range": [str(test_time_start), str(test_time_end)]
                })
                
                logger.info(f"Train time range: {train_time_str}")
                logger.info(f"  Samples: {X_train.shape[0]} ({X_train.shape[0]/n*100:.1f}%)")
                logger.info(f"Validation time range: {val_time_str}")
                logger.info(f"  Samples: {X_val.shape[0]} ({X_val.shape[0]/n*100:.1f}%)")
                logger.info(f"Test time range: {test_time_str}")
                logger.info(f"  Samples: {X_test.shape[0]} ({X_test.shape[0]/n*100:.1f}%)")
                logger.info("=" * 70)
                logger.info("✅ Time-based split confirmed: Train -> Validation -> Test (chronological order)")
                logger.info("=" * 70)
            except Exception as e:
                logger.warning(f"Failed to compute split time ranges: {e}")
                logger.warning("⚠️  Time ranges not available, but split is still time-based (using index order)")
        else:
            logger.warning("⚠️  Time index not available - cannot display time ranges")
            logger.warning("   Split is still time-based (using data index order)")
            logger.warning("   To see time ranges, ensure 'timestamp' column exists in labeled_data.parquet")

        return X_train, X_val, X_test, y_train_long, y_val_long, y_test_long, y_train_short, y_val_short, y_test_short

    def get_default_xgboost_params(self) -> dict:
        """Get default XGBoost parameters for binary classification."""
        return {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'random_state': 42,
            'n_estimators': 500,
            'early_stopping_rounds': 10,
            'tree_method': 'hist',
            'n_jobs': -1,
            'verbosity': 1
        }

    def get_preset_params(self, preset: str) -> dict:
        """Return parameter presets for reproducible training."""
        key = (preset or "").strip().lower()
        if key in ("", "default"):
            return self.get_default_xgboost_params()
        if key in ("binance_20260120", "binance_20260120_181047", "binance_good", "binance"):
            params = self.get_default_xgboost_params()
            params.update({
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            })
            return params
        if key in ("eth_binance_20260116", "eth_binance", "eth_20260116"):
            params = self.get_default_xgboost_params()
            params.update({
                'min_child_weight': 5
            })
            return params

        logger.warning(f"Unknown preset '{preset}', falling back to default params.")
        return self.get_default_xgboost_params()

    def calculate_class_weights(self, y: pd.Series) -> float:
        """Calculate scale_pos_weight for imbalanced classes."""
        n_negative = (y == 0).sum()
        n_positive = (y == 1).sum()
        if n_positive > 0:
            return n_negative / n_positive
        return 1.0

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   model_type: str = "LONG",
                   params: dict = None) -> xgb.XGBClassifier:
        """Train XGBoost model with validation."""
        logger.info(f"Training {model_type} XGBoost model...")

        if params is None:
            params = self.get_default_xgboost_params()

        # Calculate class weights
        scale_pos_weight = self.calculate_class_weights(y_train)
        params['scale_pos_weight'] = scale_pos_weight
        logger.info(f"Class weight (scale_pos_weight): {scale_pos_weight:.2f}")

        logger.info(f"Using parameters: {params}")

        # Create model
        model = xgb.XGBClassifier(**params)

        # Train with evaluation set
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        # Log training results
        results = model.evals_result()
        train_auc = results['validation_0']['auc'][-1]
        val_auc = results['validation_1']['auc'][-1]
        train_logloss = results['validation_0']['logloss'][-1]
        val_logloss = results['validation_1']['logloss'][-1]

        logger.info(f"{model_type} Training completed:")
        logger.info(f"  Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        logger.info(f"  Train LogLoss: {train_logloss:.4f}, Val LogLoss: {val_logloss:.4f}")
        logger.info(f"  Best iteration: {model.best_iteration}")

        if model_type == "LONG":
            self.model_long = model
            self.best_params_long = params
        else:
            self.model_short = model
            self.best_params_short = params

        return model

    def evaluate_model(self, model: xgb.XGBClassifier,
                      X_test: pd.DataFrame, y_test: pd.Series,
                      model_type: str = "LONG") -> dict:
        """Evaluate model performance on test set with threshold calibration."""
        logger.info(f"Evaluating {model_type} model performance...")

        # Make predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= DEFAULT_DECISION_THRESHOLD).astype(int)
        logger.info(f"Using decision threshold: {DEFAULT_DECISION_THRESHOLD:.2f}")

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        # Threshold calibration (favor precision with minimum recall)
        best_precision = -1
        best_precision_threshold = DEFAULT_DECISION_THRESHOLD
        best_f1 = -1
        best_f1_threshold = DEFAULT_DECISION_THRESHOLD
        for th in [i / 100 for i in range(40, 71)]:
            pred = (y_pred_proba >= th).astype(int)
            try:
                prec = precision_score(y_test, pred, average='binary', zero_division=0)
                rec = recall_score(y_test, pred, average='binary', zero_division=0)
                f1 = f1_score(y_test, pred, average='binary', zero_division=0)
            except Exception:
                continue

            if rec >= 0.55 and prec > best_precision:
                best_precision = prec
                best_precision_threshold = th

            if f1 > best_f1:
                best_f1 = f1
                best_f1_threshold = th

        metrics['best_precision_threshold'] = best_precision_threshold
        metrics['best_precision'] = best_precision
        metrics['best_f1_threshold'] = best_f1_threshold
        metrics['best_f1'] = best_f1

        # Log metrics
        logger.info(f"{model_type} Test Set Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        logger.info(f"{model_type} Best precision threshold (recall>=0.55): {best_precision_threshold:.2f} (prec={best_precision:.4f})")
        logger.info(f"{model_type} Best F1 threshold: {best_f1_threshold:.2f} (f1={best_f1:.4f})")

        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"{model_type} Confusion Matrix:")
        logger.info(f"  True Negatives: {cm[0,0]}")
        logger.info(f"  False Positives: {cm[0,1]}")
        logger.info(f"  False Negatives: {cm[1,0]}")
        logger.info(f"  True Positives: {cm[1,1]}")

        return metrics, cm

    def feature_importance_analysis(self, model: xgb.XGBClassifier, model_type: str = "LONG") -> pd.DataFrame:
        """Analyze and log feature importance."""
        logger.info(f"Analyzing {model_type} feature importance...")

        # Get feature importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Log top features
        logger.info(f"{model_type} Top 15 Most Important Features:")
        for idx, row in feature_importance.head(15).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return feature_importance

    def cross_validation(self, X: pd.DataFrame, y: pd.Series,
                        model_type: str = "LONG",
                        cv_folds: int = 5, params: dict = None) -> dict:
        """Perform time series cross validation."""
        logger.info(f"Performing {model_type} cross validation...")

        if params is None:
            params = self.get_default_xgboost_params()

        # Create time series split
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        # Create a copy of params without early_stopping_rounds for CV
        cv_params = params.copy()
        cv_params.pop('early_stopping_rounds', None)
        cv_params['n_estimators'] = 50  # Use fewer trees for CV to speed up
        scale_pos_weight = self.calculate_class_weights(y)
        cv_params['scale_pos_weight'] = scale_pos_weight

        # Perform cross validation
        model = xgb.XGBClassifier(**cv_params)

        # Cross validate on AUC
        cv_scores = cross_val_score(
            model, X, y, cv=tscv, scoring='roc_auc', n_jobs=-1
        )

        cv_results = {
            'mean_auc': cv_scores.mean(),
            'std_auc': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }

        logger.info(f"{model_type} Cross Validation Results ({cv_folds} folds):")
        logger.info(f"  Mean AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
        logger.info(f"  Scores: {[f'{score:.4f}' for score in cv_scores]}")

        return cv_results

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series,
                            base_params: dict = None,
                            model_type: str = "LONG") -> dict:
        """Basic hyperparameter tuning with grid search."""
        logger.info(f"Performing {model_type} hyperparameter tuning...")

        base_params = base_params.copy() if base_params is not None else self.get_default_xgboost_params()
        
        # Calculate class weights
        scale_pos_weight = self.calculate_class_weights(y_train)
        base_params['scale_pos_weight'] = scale_pos_weight

        # Note: Hyperparameter tuning is computationally expensive
        # For production, consider using --skip-tuning flag
        logger.info("Hyperparameter tuning enabled; performing grid search...")

        best_auc = 0
        best_params = base_params.copy()

        # Define parameter grid (limited for speed)
        param_grid = {
            'learning_rate': [0.03, 0.05, 0.1],
            'max_depth': [5, 6, 8],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
        }

        # Simple grid search (limited combinations for speed)
        total_combinations = len(param_grid['learning_rate']) * len(param_grid['max_depth']) * \
                           len(param_grid['min_child_weight']) * len(param_grid['subsample']) * \
                           len(param_grid['colsample_bytree'])
        logger.info(f"Testing {total_combinations} parameter combinations...")

        combination_count = 0
        for lr in param_grid['learning_rate']:
            for md in param_grid['max_depth']:
                for mcw in param_grid['min_child_weight']:
                    for ss in param_grid['subsample']:
                        for csbt in param_grid['colsample_bytree']:
                            combination_count += 1
                            test_params = base_params.copy()
                            test_params.update({
                                'learning_rate': lr,
                                'max_depth': md,
                                'min_child_weight': mcw,
                                'subsample': ss,
                                'colsample_bytree': csbt
                            })

                            # Train model
                            model = xgb.XGBClassifier(**test_params)
                            model.fit(
                                X_train, y_train,
                                eval_set=[(X_val, y_val)],
                                verbose=False
                            )

                            # Evaluate
                            val_pred_proba = model.predict_proba(X_val)[:, 1]
                            val_auc = roc_auc_score(y_val, val_pred_proba)

                            if val_auc > best_auc:
                                best_auc = val_auc
                                best_params = test_params.copy()
                                logger.info(f"  [{combination_count}/{total_combinations}] New best AUC: {best_auc:.4f}")

        logger.info(f"{model_type} Best validation AUC: {best_auc:.4f}")
        logger.info(f"{model_type} Best parameters: {best_params}")

        return best_params

    def save_model_and_results(self, model_long: xgb.XGBClassifier, model_short: xgb.XGBClassifier,
                             metrics_long: dict, metrics_short: dict,
                             cv_results_long: dict, cv_results_short: dict,
                             feature_importance_long: pd.DataFrame, feature_importance_short: pd.DataFrame):
        """Save models and training results."""
        logger.info("Saving models and results...")

        # Create timestamp for model version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create models directory if it doesn't exist
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)

        # Save LONG model
        model_long_name = f"model_long_{timestamp}.joblib"
        model_long_path = models_dir / model_long_name
        joblib.dump(model_long, model_long_path)
        logger.info(f"LONG model saved to {model_long_path}")

        # Save SHORT model
        model_short_name = f"model_short_{timestamp}.joblib"
        model_short_path = models_dir / model_short_name
        joblib.dump(model_short, model_short_path)
        logger.info(f"SHORT model saved to {model_short_path}")

        # Save as latest models
        latest_long_path = models_dir / 'latest_model_long.joblib'
        latest_short_path = models_dir / 'latest_model_short.joblib'
        joblib.dump(model_long, latest_long_path)
        joblib.dump(model_short, latest_short_path)
        logger.info(f"Latest models saved")

        # Save training results
        results = {
            'timestamp': timestamp,
            'model_long_name': model_long_name,
            'model_short_name': model_short_name,
            'parameters_long': self.best_params_long,
            'parameters_short': self.best_params_short,
            'metrics_long': metrics_long,
            'metrics_short': metrics_short,
            'cross_validation_long': cv_results_long,
            'cross_validation_short': cv_results_short,
            'feature_count': len(self.feature_names),
            'sample_count': self.sample_count,
            'split_info': self.split_info or {},
            'feature_names': self.feature_names
        }

        results_path = self.output_dir / f'training_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

        # Save threshold recommendations for QuantConnect usage
        thresholds_path = self.output_dir / 'model_thresholds.json'
        try:
            thresholds_payload = {
                "long": {
                    "best_precision_threshold": metrics_long.get("best_precision_threshold"),
                    "best_precision": metrics_long.get("best_precision"),
                    "best_f1_threshold": metrics_long.get("best_f1_threshold"),
                    "best_f1": metrics_long.get("best_f1"),
                },
                "short": {
                    "best_precision_threshold": metrics_short.get("best_precision_threshold"),
                    "best_precision": metrics_short.get("best_precision"),
                    "best_f1_threshold": metrics_short.get("best_f1_threshold"),
                    "best_f1": metrics_short.get("best_f1"),
                }
            }
            with open(thresholds_path, 'w') as f:
                json.dump(thresholds_payload, f, indent=2)
            logger.info(f"Thresholds saved to {thresholds_path}")
        except Exception as e:
            logger.warning(f"Failed to save thresholds: {e}")

        # Save feature importance
        importance_long_path = self.output_dir / f'feature_importance_long_{timestamp}.csv'
        feature_importance_long.to_csv(importance_long_path, index=False)
        logger.info(f"LONG feature importance saved to {importance_long_path}")

        importance_short_path = self.output_dir / f'feature_importance_short_{timestamp}.csv'
        feature_importance_short.to_csv(importance_short_path, index=False)
        logger.info(f"SHORT feature importance saved to {importance_short_path}")

        # Save feature names
        features_path = self.output_dir / 'model_features.txt'
        with open(features_path, 'w') as f:
            f.write("Model Features:\n")
            f.write("=" * 20 + "\n")
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        logger.info(f"Feature list saved to {features_path}")

def main():
    """Main training function."""
    args = parse_arguments()
    validate_arguments(args)

    # Set up logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create data filter
    data_filter = DataFilter(args)

    # Initialize trainer
    trainer = XGBoostTrainer(data_filter, args.output_dir)

    try:
        # Load training data
        logger.info("Loading training data...")
        X, y_long, y_short = trainer.load_training_data()

        # Prepare data splits (train/val/test)
        X_train, X_val, X_test, y_train_long, y_val_long, y_test_long, y_train_short, y_val_short, y_test_short = trainer.prepare_data_splits(
            X, y_long, y_short
        )

        # Parameter preset (e.g., binance_20260120) + optional tuning
        preset = args.xgb_preset if hasattr(args, 'xgb_preset') else 'default'
        if (preset or "").strip().lower() in ("", "default"):
            # Auto-select preset based on model version if available
            model_version = args.model_version if hasattr(args, 'model_version') else ''
            if "bybit" in str(model_version).lower():
                preset = "eth_binance_20260116"
                logger.info("Auto-select preset for Bybit: eth_binance_20260116")
        
        base_params_long = trainer.get_preset_params(preset)
        base_params_short = trainer.get_preset_params(preset)
        logger.info(f"Using XGBoost preset: {preset}")

        # Hyperparameter tuning
        if args.skip_tuning:
            logger.info("Skipping hyperparameter tuning; using preset/base params.")
            best_params_long = base_params_long
            best_params_short = base_params_short
        else:
            logger.info("Performing hyperparameter tuning for LONG model...")
            best_params_long = trainer.hyperparameter_tuning(
                X_train, y_train_long, X_val, y_val_long, 
                base_params=base_params_long, model_type="LONG"
            )
            logger.info("Performing hyperparameter tuning for SHORT model...")
            best_params_short = trainer.hyperparameter_tuning(
                X_train, y_train_short, X_val, y_val_short,
                base_params=base_params_short, model_type="SHORT"
            )

        # Train LONG model
        logger.info("Training LONG model...")
        model_long = trainer.train_model(
            X_train, y_train_long, X_val, y_val_long, 
            model_type="LONG", params=best_params_long
        )

        # Train SHORT model
        logger.info("Training SHORT model...")
        model_short = trainer.train_model(
            X_train, y_train_short, X_val, y_val_short,
            model_type="SHORT", params=best_params_short
        )

        # Evaluate models
        metrics_long, cm_long = trainer.evaluate_model(model_long, X_test, y_test_long, model_type="LONG")
        metrics_short, cm_short = trainer.evaluate_model(model_short, X_test, y_test_short, model_type="SHORT")

        # Feature importance
        feature_importance_long = trainer.feature_importance_analysis(model_long, model_type="LONG")
        feature_importance_short = trainer.feature_importance_analysis(model_short, model_type="SHORT")

        # Cross validation
        cv_results_long = trainer.cross_validation(X, y_long, model_type="LONG")
        cv_results_short = trainer.cross_validation(X, y_short, model_type="SHORT")

        # Save everything
        trainer.save_model_and_results(
            model_long, model_short,
            metrics_long, metrics_short,
            cv_results_long, cv_results_short,
            feature_importance_long, feature_importance_short
        )

        logger.info("\n=== Training Complete ===")
        logger.info("Ready for model_uploader.py")

    except Exception as e:
        logger.error(f"Error in training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
