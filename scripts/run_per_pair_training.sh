#!/bin/bash
# Train RL policy for a single currency pair
#
# Usage:
#   bash scripts/run_per_pair_training.sh EURUSD
#   bash scripts/run_per_pair_training.sh BTCUSD --gpu
#   bash scripts/run_per_pair_training.sh GBPUSD --epochs 100 --episodes-per-epoch 200

set -e  # Exit on any error

# Save project root for absolute paths
PROJECT_ROOT=$(pwd)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/run_per_pair_training.sh <PAIR> [--gpu] [--epochs N] [--episodes-per-epoch N]"
    echo ""
    echo "Examples:"
    echo "  bash scripts/run_per_pair_training.sh EURUSD"
    echo "  bash scripts/run_per_pair_training.sh BTCUSD --gpu --epochs 100"
    echo ""
    exit 1
fi

# Extract pair from first argument
PAIR=$1
shift

# Configuration defaults
EPOCHS=100
EPISODES_PER_EPOCH=200
SEED=42
USE_GPU=false

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            USE_GPU=true
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --episodes-per-epoch)
            EPISODES_PER_EPOCH="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert pair to lowercase for directory names
PAIR_LOWER=$(echo "$PAIR" | tr '[:upper:]' '[:lower:]')

# Create log directory with absolute path
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/training_logs/$PAIR_LOWER"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}" | tee -a "$LOG_FILE"
}

# Header
echo ""
echo "================================================================================"
echo "  RL TRAINING - $PAIR"
echo "================================================================================"
echo "  Pair: $PAIR"
echo "  Epochs: $EPOCHS"
echo "  Episodes per epoch: $EPISODES_PER_EPOCH"
echo "  Total episodes: $((EPOCHS * EPISODES_PER_EPOCH))"
echo "  GPU: $USE_GPU"
echo "  Seed: $SEED"
echo "  Log: $LOG_FILE"
echo "================================================================================"
echo ""

# Check we're in the right directory
if [ ! -f "scripts/train_on_historical_data.py" ]; then
    log_error "Not in project root directory. Please run from /Volumes/Containers/Sequence"
    exit 1
fi

# Check if prepared data exists
DATA_DIR="data/prepared/$PAIR_LOWER"
if [ ! -d "$DATA_DIR" ]; then
    log_error "Prepared data not found for $PAIR at: $DATA_DIR"
    log "Please run data collection first:"
    log "  bash scripts/run_data_collection.sh"
    exit 1
fi

# Count prepared files
NPY_COUNT=$(ls -1 "$DATA_DIR"/*.npy 2>/dev/null | wc -l | tr -d ' ')
if [ "$NPY_COUNT" -eq 0 ]; then
    log_error "No .npy files found in $DATA_DIR"
    exit 1
fi

log "Found $NPY_COUNT prepared .npy files for $PAIR"

#------------------------------------------------------------------------------
# TRAINING
#------------------------------------------------------------------------------
log "Starting RL training for $PAIR..."
log "Training configuration:"
log "  - Epochs: $EPOCHS"
log "  - Episodes per epoch: $EPISODES_PER_EPOCH"
log "  - Total episodes: $((EPOCHS * EPISODES_PER_EPOCH))"
log "  - GPU: $USE_GPU"
log "  - Seed: $SEED"

# Build training command
TRAIN_CMD="SEQ_LOG_LEVEL=INFO python3 scripts/train_on_historical_data.py \
    --data-dir $DATA_DIR \
    --epochs $EPOCHS \
    --episodes-per-epoch $EPISODES_PER_EPOCH \
    --output-dir checkpoints/$PAIR_LOWER/production_${TIMESTAMP} \
    --seed $SEED"

if [ "$USE_GPU" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --gpu"
fi

# Run training
log "Starting training (estimated time: 2-4 hours on CPU, 30-60 min on GPU)..."
log "You can monitor progress in: $LOG_FILE"

START_TIME=$(date +%s)

if eval "$TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))

    log_success "Training complete for $PAIR!"
    log "Training duration: ${HOURS}h ${MINUTES}m"
else
    log_error "Training failed for $PAIR. Check log: $LOG_FILE"
    exit 1
fi

#------------------------------------------------------------------------------
# COMPLETION SUMMARY
#------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "  ✅ TRAINING COMPLETE - $PAIR"
echo "================================================================================"
echo ""
log_success "$PAIR model training finished successfully!"
echo ""
echo "Results:"
echo "  - Model checkpoints: checkpoints/$PAIR_LOWER/production_${TIMESTAMP}/"
echo "  - Training log: $LOG_FILE"
echo "  - Training duration: ${HOURS}h ${MINUTES}m"
echo ""
echo "Next steps:"
echo "  1. Review training metrics in log file"
echo "  2. Load the trained policy:"
echo "     checkpoints/$PAIR_LOWER/production_${TIMESTAMP}/policy_epoch_*.pt"
echo "  3. Backtest on held-out data"
echo "  4. Deploy to paper trading when ready"
echo ""
echo "================================================================================"
