#!/bin/bash

# Simple script untuk menjalankan pipeline AlphaTrade XGBoost step by step

echo "========================================"
echo "AlphaTrade XGBoost Pipeline - Manual Run"
echo "========================================"

# Default parameters
EXCHANGE=${EXCHANGE:-binance}
PAIR=${PAIR:-btcusdt.p}
INTERVAL=${INTERVAL:-1h}
DB_PATH=${DB_PATH:-../sqlite_crypto.db}
OUTPUT_DIR=${OUTPUT_DIR:-./output_train}
TP_PCT=${TP_PCT:-0.02}
SL_PCT=${SL_PCT:-0.01}
HORIZON_HOURS=${HORIZON_HOURS:-24}
MODEL_VERSION=${MODEL_VERSION:-}

echo "Configuration:"
echo "  Exchange: $EXCHANGE"
echo "  Pair: $PAIR"
echo "  Interval: $INTERVAL"
echo "  Database: $DB_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  TP: ${TP_PCT} (${TP_PCT}%)"
echo "  SL: ${SL_PCT} (${SL_PCT}%)"
echo "  Horizon: ${HORIZON_HOURS} hours"
if [ -n "$MODEL_VERSION" ]; then
    echo "  Model Version: $MODEL_VERSION"
fi
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run a step
run_step() {
    local script=$1
    local step_name=$2
    shift 2  # Remove first two arguments, remaining are flags
    local extra_flags="$@"  # Capture all remaining flags

    echo "=========================================="
    echo "Running $step_name..."
    echo "Command: python $script $extra_flags --exchange $EXCHANGE --pair $PAIR --interval $INTERVAL --db-path $DB_PATH --output-dir $OUTPUT_DIR --tp-pct $TP_PCT --sl-pct $SL_PCT --horizon-hours $HORIZON_HOURS"
    echo "=========================================="

    if python "$script" $extra_flags \
        --exchange "$EXCHANGE" \
        --pair "$PAIR" \
        --interval "$INTERVAL" \
        --db-path "$DB_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --tp-pct "$TP_PCT" \
        --sl-pct "$SL_PCT" \
        --horizon-hours "$HORIZON_HOURS"; then
        echo "✅ $step_name completed successfully"
        echo ""
    else
        echo "❌ $step_name failed!"
        echo "Pipeline stopped."
        exit 1
    fi
}

# Parse extra flags (optional: --days N, --time start,end, --upload-to-api, --model-version, --xgb-preset)
EXTRA_FLAGS=""
UPLOAD_FLAGS=""
XGB_FLAGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --days)
            EXTRA_FLAGS="$EXTRA_FLAGS --days $2"
            shift 2
            ;;
        --time)
            EXTRA_FLAGS="$EXTRA_FLAGS --time $2"
            shift 2
            ;;
        --upload-to-api)
            UPLOAD_FLAGS="$UPLOAD_FLAGS --upload-to-api"
            shift
            ;;
        --model-version)
            MODEL_VERSION="$2"
            UPLOAD_FLAGS="$UPLOAD_FLAGS --model-version $2"
            shift 2
            ;;
        --api-domain)
            UPLOAD_FLAGS="$UPLOAD_FLAGS --api-domain $2"
            shift 2
            ;;
        --xgb-preset)
            XGB_FLAGS="$XGB_FLAGS --xgb-preset $2"
            shift 2
            ;;
        --skip-tuning)
            XGB_FLAGS="$XGB_FLAGS --skip-tuning"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Run each step
run_step "load_database.py" "Step 1: Load Database" $EXTRA_FLAGS
run_step "feature_engineering.py" "Step 2: Feature Engineering" $EXTRA_FLAGS
run_step "label_builder.py" "Step 3: Label Building" $EXTRA_FLAGS
run_step "xgboost_trainer.py" "Step 4: Model Training" $EXTRA_FLAGS $XGB_FLAGS

# Upload models if requested
if [ -n "$UPLOAD_FLAGS" ] || [ -n "$MODEL_VERSION" ]; then
    echo "=========================================="
    echo "Running Step 5: Model Upload"
    echo "=========================================="
    
    UPLOAD_CMD="python model_uploader.py --exchange $EXCHANGE --pair $PAIR --interval $INTERVAL --db-path $DB_PATH --output-dir $OUTPUT_DIR"
    
    if [ -n "$MODEL_VERSION" ]; then
        UPLOAD_CMD="$UPLOAD_CMD --model-version $MODEL_VERSION"
    fi
    
    UPLOAD_CMD="$UPLOAD_CMD $UPLOAD_FLAGS"
    
    echo "Command: $UPLOAD_CMD"
    echo "=========================================="
    
    if eval "$UPLOAD_CMD"; then
        echo "✅ Model upload completed successfully"
        echo ""
    else
        echo "❌ Model upload failed!"
        echo "Pipeline completed but upload failed."
    fi
else
    echo "=========================================="
    echo "Skipping Model Upload (use --upload-to-api to enable)"
    echo "=========================================="
fi

echo "=========================================="
echo "✅ Pipeline completed successfully!"
echo "=========================================="

# Show structured output directory contents
echo ""
echo "Structured Output Directory Contents:"
echo "📁 $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR"

echo ""
echo "📁 Model files:"
if [ -d "$OUTPUT_DIR/models" ]; then
    ls -la "$OUTPUT_DIR/models/"
else
    ls -la "$OUTPUT_DIR"/*.joblib 2>/dev/null || echo "No model files found"
fi

echo ""
echo "📁 Dataset files:"
if [ -d "$OUTPUT_DIR/datasets" ]; then
    echo "  📄 Dataset summary:"
    ls -la "$OUTPUT_DIR/datasets/"*summary*.txt 2>/dev/null || echo "  No dataset summary found"
    echo ""
    echo "  📄 Datasets:"
    ls -la "$OUTPUT_DIR/datasets/"*.parquet 2>/dev/null || echo "  No dataset files found"
fi

echo ""
echo "📁 Feature files:"
if [ -d "$OUTPUT_DIR/features" ]; then
    ls -la "$OUTPUT_DIR/features/" || echo "No feature files found"
fi

echo ""
echo "Next steps:"
echo "1. Models are ready for QuantConnect deployment"
echo "2. Use --upload-to-api to upload models to server"
echo "3. QuantConnect will auto-download models when version changes"
echo ""
echo "✅ Pipeline completed with structured output!"
