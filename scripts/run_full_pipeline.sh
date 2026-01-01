#!/bin/bash
# Complete FX RL Training Pipeline
# This script runs the full data collection, preprocessing, and training pipeline
#
# Usage:
#   bash scripts/run_full_pipeline.sh
#   bash scripts/run_full_pipeline.sh --gpu  # For GPU training

set -e  # Exit on any error

# Save project root for absolute paths
PROJECT_ROOT=$(pwd)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
START_YEAR=2014
END_YEAR=2024
MAX_DOWNLOADS=
PAIR="EURUSD"
SEQUENCE_LENGTH=390
EPOCHS=100
EPISODES_PER_EPOCH=200
SEED=42

# Parse arguments
USE_GPU=false
if [[ "$1" == "--gpu" ]]; then
    USE_GPU=true
fi

# Create log directory with absolute path
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/training_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/full_pipeline_${TIMESTAMP}.log"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

# Header
echo ""
echo "================================================================================"
echo "  FX RL TRAINING PIPELINE - COMPLETE DATA COLLECTION & TRAINING"
echo "================================================================================"
echo "  Pair: $PAIR"
echo "  Years: $START_YEAR - $END_YEAR"
echo "  Training: $EPOCHS epochs × $EPISODES_PER_EPOCH episodes = $((EPOCHS * EPISODES_PER_EPOCH)) total episodes"
echo "  GPU: $USE_GPU"
echo "  Log: $LOG_FILE"
echo "================================================================================"
echo ""

# Check we're in the right directory
if [ ! -f "scripts/train_on_historical_data.py" ]; then
    log_error "Not in project root directory. Please run from /Volumes/Containers/Sequence"
    exit 1
fi

#------------------------------------------------------------------------------
# PHASE 1: DATA DOWNLOAD
#------------------------------------------------------------------------------
log "PHASE 1/4: Downloading historical FX data..."
log "Downloading data from $START_YEAR to $END_YEAR (max $MAX_DOWNLOADS downloads)"

if SEQ_LOG_LEVEL=INFO python3 data/download_all_fx_data.py \
    --start-year $START_YEAR \
    --end-year $END_YEAR \
    --max-downloads $MAX_DOWNLOADS \
    --output data/histdata >> "$LOG_FILE" 2>&1; then
    log_success "Data download complete"
else
    log_error "Data download failed. Check log: $LOG_FILE"
    exit 1
fi

#------------------------------------------------------------------------------
# PHASE 2: EXTRACT DATA
#------------------------------------------------------------------------------
log "PHASE 2/4: Extracting ZIP files..."

PAIR_LOWER=$(echo "$PAIR" | tr '[:upper:]' '[:lower:]')
DATA_DIR="data/histdata/$PAIR_LOWER"

if [ ! -d "$DATA_DIR" ]; then
    log_error "Data directory not found: $DATA_DIR"
    exit 1
fi

cd "$DATA_DIR"
if unzip -o "*.zip" >> "$LOG_FILE" 2>&1; then
    log_success "ZIP extraction complete"
else
    log_warning "Some ZIP files may have failed to extract (this is OK if CSVs exist)"
fi
cd - > /dev/null

# Count CSV files
CSV_COUNT=$(ls -1 "$DATA_DIR"/*.csv 2>/dev/null | wc -l | tr -d ' ')
log "Found $CSV_COUNT CSV files"

if [ "$CSV_COUNT" -eq 0 ]; then
    log_error "No CSV files found after extraction"
    exit 1
fi

#------------------------------------------------------------------------------
# PHASE 3: VALIDATE & PREPARE DATA
#------------------------------------------------------------------------------
log "PHASE 3/4: Validating and preparing data for training..."
log "Preparing sequences of length $SEQUENCE_LENGTH"

if SEQ_LOG_LEVEL=INFO python3 scripts/validate_training_data.py \
    --pair "$PAIR" \
    --data-root data \
    --prepare \
    --output data/prepared \
    --sequence-length $SEQUENCE_LENGTH >> "$LOG_FILE" 2>&1; then
    log_success "Data preparation complete"
else
    log_error "Data preparation failed. Check log: $LOG_FILE"
    exit 1
fi

# Count prepared files
NPY_COUNT=$(ls -1 data/prepared/*.npy 2>/dev/null | wc -l | tr -d ' ')
log "Prepared $NPY_COUNT .npy files for training"

if [ "$NPY_COUNT" -eq 0 ]; then
    log_error "No .npy files created"
    exit 1
fi

#------------------------------------------------------------------------------
# PHASE 4: TRAINING
#------------------------------------------------------------------------------
log "PHASE 4/4: Starting RL training..."
log "Training configuration:"
log "  - Epochs: $EPOCHS"
log "  - Episodes per epoch: $EPISODES_PER_EPOCH"
log "  - Total episodes: $((EPOCHS * EPISODES_PER_EPOCH))"
log "  - GPU: $USE_GPU"
log "  - Seed: $SEED"

# Build training command
TRAIN_CMD="SEQ_LOG_LEVEL=INFO python3 scripts/train_on_historical_data.py \
    --epochs $EPOCHS \
    --episodes-per-epoch $EPISODES_PER_EPOCH \
    --output-dir checkpoints/production_${TIMESTAMP} \
    --seed $SEED"

if [ "$USE_GPU" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --gpu"
fi

# Run training
log "Starting training (this will take 2-4 hours on CPU, 30-60 min on GPU)..."
log "You can monitor progress in: $LOG_FILE"

START_TIME=$(date +%s)

if eval "$TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))

    log_success "Training complete!"
    log "Training duration: ${HOURS}h ${MINUTES}m"
else
    log_error "Training failed. Check log: $LOG_FILE"
    exit 1
fi

#------------------------------------------------------------------------------
# COMPLETION SUMMARY
#------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "  ✅ PIPELINE COMPLETE"
echo "================================================================================"
echo ""
log_success "All phases completed successfully!"
echo ""
echo "Results:"
echo "  - Prepared data: data/prepared/*.npy ($NPY_COUNT files)"
echo "  - Model checkpoints: checkpoints/production_${TIMESTAMP}/"
echo "  - Training log: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Review training metrics in the log file"
echo "  2. Load the trained policy: checkpoints/production_${TIMESTAMP}/policy_epoch_*.pt"
echo "  3. Backtest the policy on held-out data"
echo "  4. Deploy to paper trading when ready"
echo ""
echo "================================================================================"