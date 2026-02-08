#!/usr/bin/env python3
"""
Upload trained models to API server for QuantConnect integration.
Supports uploading both LONG and SHORT models.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging
import joblib
import base64
import json
import requests
from datetime import datetime
from typing import Optional
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

class ModelUploader:
    """Upload models to API server."""

    def __init__(self, data_filter: DataFilter, output_dir: str = './output_train',
                 api_domain: str = 'https://api.dragonfortune.ai',
                 api_version: str = 'v1'):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.api_domain = api_domain.rstrip('/')
        self.api_version = api_version
        self.model_version = None

    def load_latest_models(self) -> tuple:
        """Load latest trained models."""
        logger.info("Loading latest models...")

        models_dir = self.output_dir / 'models'
        model_long_path = models_dir / 'latest_model_long.joblib'
        model_short_path = models_dir / 'latest_model_short.joblib'

        if not model_long_path.exists() or not model_short_path.exists():
            logger.error("Latest models not found. Run xgboost_trainer.py first.")
            sys.exit(1)

        try:
            model_long = joblib.load(model_long_path)
            model_short = joblib.load(model_short_path)
            logger.info("Models loaded successfully")
            return model_long, model_short
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            sys.exit(1)

    def load_feature_names(self) -> list:
        """Load feature names from training."""
        feature_file = self.output_dir / 'model_features.txt'
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                lines = f.readlines()
                # Skip header lines
                features = [line.strip() for line in lines if line.strip() and not line.startswith('=') and not line.startswith('Model')]
                return features
        
        # Fallback: try to get from model
        try:
            models_dir = self.output_dir / 'models'
            model_long_path = models_dir / 'latest_model_long.joblib'
            model = joblib.load(model_long_path)
            if hasattr(model, 'feature_names_in_'):
                return list(model.feature_names_in_)
        except:
            pass

        logger.warning("Could not load feature names, using empty list")
        return []

    def encode_model_to_base64(self, model_path: Path) -> str:
        """Encode model file to base64 string."""
        try:
            with open(model_path, 'rb') as f:
                model_bytes = f.read()
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
            logger.info(f"Encoded model: {len(model_bytes)} bytes -> {len(model_b64)} base64 chars")
            return model_b64
        except Exception as e:
            logger.error(f"Error encoding model: {e}")
            raise

    def upload_model(self, model_path: Path, model_type: str, feature_names: list) -> bool:
        """Upload a single model to API server."""
        logger.info(f"Uploading {model_type} model to API...")

        # Generate model version if not set
        if not self.model_version:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pair = self.data_filter.pair.replace('.', '_').upper()
            self.model_version = f"alphatrade_{pair}_{timestamp}"

        # Encode model
        model_b64 = self.encode_model_to_base64(model_path)

        # Prepare payload
        payload = {
            "model_version": f"{self.model_version}_{model_type.lower()}",
            "model_data_base64": model_b64,
            "feature_names": feature_names,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": model_type
        }

        # API endpoint
        api_url = f"{self.api_domain}/api/{self.api_version}/alphatrade/latest/model"
        
        # Headers
        headers = {
            "Content-Type": "application/json"
        }

        try:
            logger.info(f"Uploading to: {api_url}")
            response = requests.post(api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()

            result = response.json()
            if result.get("success"):
                logger.info(f"✅ {model_type} model uploaded successfully")
                logger.info(f"   Model version: {result.get('model_version', 'N/A')}")
                return True
            else:
                logger.error(f"❌ Upload failed: {result}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Upload error: {e}")
            return False

    def upload_models(self) -> bool:
        """Upload both LONG and SHORT models to API."""
        logger.info("Starting model upload process...")

        # Load models
        model_long, model_short = self.load_latest_models()

        # Load feature names
        feature_names = self.load_feature_names()

        # Get model paths
        models_dir = self.output_dir / 'models'
        model_long_path = models_dir / 'latest_model_long.joblib'
        model_short_path = models_dir / 'latest_model_short.joblib'

        # Upload LONG model
        success_long = self.upload_model(model_long_path, "LONG", feature_names)

        # Upload SHORT model
        success_short = self.upload_model(model_short_path, "SHORT", feature_names)

        if success_long and success_short:
            logger.info("✅ Both models uploaded successfully!")
            return True
        else:
            logger.error("❌ Some models failed to upload")
            return False

def main():
    """Main upload function."""
    args = parse_arguments()
    validate_arguments(args)

    if not args.upload_to_api:
        logger.info("Upload to API disabled. Use --upload-to-api to enable.")
        return

    # Set up logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create data filter
    data_filter = DataFilter(args)

    # Initialize uploader
    uploader = ModelUploader(
        data_filter,
        args.output_dir,
        api_domain=args.api_domain,
        api_version=args.api_version
    )

    # Set model version if provided
    if args.model_version:
        uploader.model_version = args.model_version

    try:
        # Upload models
        success = uploader.upload_models()

        if success:
            logger.info("\n=== Model Upload Complete ===")
            sys.exit(0)
        else:
            logger.error("\n=== Model Upload Failed ===")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error in model upload: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
