#!/bin/bash
# Train RL policies for ALL currency pairs
# Can run sequentially or in parallel (if resources allow)
#
# Usage:
#   bash scripts/run_all_training.sh                    # Sequential training
#   bash scripts/run_all_training.sh --parallel 4       # Parallel with 4 workers
#   bash scripts/run_all_training.sh --gpu --parallel 2 # GPU parallel training

set -e  # Exit on any error

# Save project root for absolute paths
PROJECT_ROOT=$(pwd)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration defaults
EPOCHS=100
EPISODES_PER_EPOCH=200
SEED=42
USE_GPU=false
PARALLEL=1  # Number of parallel training jobs
PAIRS_CSV="$PROJECT_ROOT/pairs.csv"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            USE_GPU=true
            shift
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --episodes-per-epoch)
            EPISODES_PER_EPOCH="$2"
            shift 2
            ;;
        --pairs-csv)
            PAIRS_CSV="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create log directory with absolute path
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/training_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/all_training_${TIMESTAMP}.log"

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

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

# Header
echo ""
echo "================================================================================"
echo "  MULTI-PAIR RL TRAINING PIPELINE"
echo "================================================================================"
echo "  Pairs CSV: $PAIRS_CSV"
echo "  Epochs: $EPOCHS per pair"
echo "  Episodes per epoch: $EPISODES_PER_EPOCH"
echo "  GPU: $USE_GPU"
echo "  Parallel jobs: $PARALLEL"
echo "  Log: $LOG_FILE"
echo "================================================================================"
echo ""

# Check we're in the right directory
if [ ! -f "$PAIRS_CSV" ]; then
    log_error "Pairs CSV not found: $PAIRS_CSV"
    exit 1
fi

# Count pairs in CSV
TOTAL_PAIRS=$(tail -n +2 "$PAIRS_CSV" | wc -l | tr -d ' ')
log "Found $TOTAL_PAIRS currency pairs to train"

# Estimate training time
if [ "$USE_GPU" = true ]; then
    EST_TIME_PER_PAIR=45  # minutes on GPU
else
    EST_TIME_PER_PAIR=150  # minutes on CPU
fi

if [ "$PARALLEL" -gt 1 ]; then
    TOTAL_EST_TIME=$((TOTAL_PAIRS * EST_TIME_PER_PAIR / PARALLEL))
else
    TOTAL_EST_TIME=$((TOTAL_PAIRS * EST_TIME_PER_PAIR))
fi

EST_HOURS=$((TOTAL_EST_TIME / 60))
EST_DAYS=$((EST_HOURS / 24))

log "Estimated training time:"
log "  - Per pair: ~${EST_TIME_PER_PAIR} minutes"
log "  - Total: ~${EST_HOURS} hours (~${EST_DAYS} days)"
log "  - With $PARALLEL parallel jobs"
echo ""

# Collect pairs to train
PAIRS_TO_TRAIN=()
PAIRS_SKIPPED=0

tail -n +2 "$PAIRS_CSV" | while IFS=',' read -r currency_pair_name pair history_first_trading_month; do
    PAIR_LOWER=$(echo "$pair" | tr '[:upper:]' '[:lower:]')
    DATA_DIR="data/prepared/$PAIR_LOWER"

    # Check if prepared data exists
    if [ ! -d "$DATA_DIR" ]; then
        log_warning "Skipping $pair - no prepared data at $DATA_DIR"
        PAIRS_SKIPPED=$((PAIRS_SKIPPED + 1))
        continue
    fi

    NPY_COUNT=$(ls -1 "$DATA_DIR"/*.npy 2>/dev/null | wc -l | tr -d ' ')
    if [ "$NPY_COUNT" -eq 0 ]; then
        log_warning "Skipping $pair - no .npy files in $DATA_DIR"
        PAIRS_SKIPPED=$((PAIRS_SKIPPED + 1))
        continue
    fi

    PAIRS_TO_TRAIN+=("$pair")
done

PAIRS_COUNT=${#PAIRS_TO_TRAIN[@]}
log "Pairs ready for training: $PAIRS_COUNT"
log "Pairs skipped (no data): $PAIRS_SKIPPED"
echo ""

if [ "$PAIRS_COUNT" -eq 0 ]; then
    log_error "No pairs ready for training. Please run data collection first:"
    log "  bash scripts/run_data_collection.sh"
    exit 1
fi

#------------------------------------------------------------------------------
# TRAINING LOOP
#------------------------------------------------------------------------------
log "Starting training for $PAIRS_COUNT pairs..."
echo ""

PAIRS_COMPLETED=0
PAIRS_FAILED=0
START_TIME=$(date +%s)

# Build GPU flag
GPU_FLAG=""
if [ "$USE_GPU" = true ]; then
    GPU_FLAG="--gpu"
fi

# Function to train a single pair
train_pair() {
    local pair=$1
    log "[$pair] Starting training..."

    if bash scripts/run_per_pair_training.sh "$pair" \
        $GPU_FLAG \
        --epochs "$EPOCHS" \
        --episodes-per-epoch "$EPISODES_PER_EPOCH" \
        --seed "$SEED" >> "$LOG_FILE" 2>&1; then

        log_success "[$pair] Training complete"
        return 0
    else
        log_error "[$pair] Training failed"
        return 1
    fi
}

if [ "$PARALLEL" -eq 1 ]; then
    # Sequential training
    log "Running sequential training (1 pair at a time)..."

    tail -n +2 "$PAIRS_CSV" | while IFS=',' read -r currency_pair_name pair history_first_trading_month; do
        PAIR_LOWER=$(echo "$pair" | tr '[:upper:]' '[:lower:]')
        DATA_DIR="data/prepared/$PAIR_LOWER"

        # Skip pairs without data
        if [ ! -d "$DATA_DIR" ]; then
            continue
        fi

        NPY_COUNT=$(ls -1 "$DATA_DIR"/*.npy 2>/dev/null | wc -l | tr -d ' ')
        if [ "$NPY_COUNT" -eq 0 ]; then
            continue
        fi

        # Train this pair
        if train_pair "$pair"; then
            PAIRS_COMPLETED=$((PAIRS_COMPLETED + 1))
        else
            PAIRS_FAILED=$((PAIRS_FAILED + 1))
        fi

        log "Progress: $PAIRS_COMPLETED/$PAIRS_COUNT completed, $PAIRS_FAILED failed"
    done

else
    # Parallel training
    log "Running parallel training ($PARALLEL pairs at a time)..."
    log_warning "Parallel training requires sufficient GPU/CPU resources!"

    # Export function for parallel execution
    export -f train_pair log log_success log_error
    export LOG_FILE EPOCHS EPISODES_PER_EPOCH SEED GPU_FLAG
    export BLUE GREEN RED NC

    # Read pairs into array
    mapfile -t ALL_PAIRS < <(tail -n +2 "$PAIRS_CSV" | cut -d',' -f2)

    # Filter to only pairs with data
    VALID_PAIRS=()
    for pair in "${ALL_PAIRS[@]}"; do
        PAIR_LOWER=$(echo "$pair" | tr '[:upper:]' '[:lower:]')
        DATA_DIR="data/prepared/$PAIR_LOWER"

        if [ -d "$DATA_DIR" ]; then
            NPY_COUNT=$(ls -1 "$DATA_DIR"/*.npy 2>/dev/null | wc -l | tr -d ' ')
            if [ "$NPY_COUNT" -gt 0 ]; then
                VALID_PAIRS+=("$pair")
            fi
        fi
    done

    # Run parallel training using xargs
    printf '%s\n' "${VALID_PAIRS[@]}" | xargs -P "$PARALLEL" -I {} bash -c 'train_pair "$@"' _ {}

    # Count results
    PAIRS_COMPLETED=$(grep -c "Training complete" "$LOG_FILE" || echo "0")
    PAIRS_FAILED=$(grep -c "Training failed" "$LOG_FILE" || echo "0")
fi

#------------------------------------------------------------------------------
# COMPLETION SUMMARY
#------------------------------------------------------------------------------
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "================================================================================"
echo "  ✅ MULTI-PAIR TRAINING COMPLETE"
echo "================================================================================"
echo ""
log_success "Training finished for all pairs!"
echo ""
echo "Results:"
echo "  - Total pairs: $TOTAL_PAIRS"
echo "  - Pairs trained: $PAIRS_COMPLETED"
echo "  - Pairs failed: $PAIRS_FAILED"
echo "  - Pairs skipped: $PAIRS_SKIPPED"
echo "  - Training duration: ${HOURS}h ${MINUTES}m"
echo "  - Model checkpoints: checkpoints/{pair}/production_*/"
echo "  - Training log: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Review training logs for each pair: training_logs/{pair}/training_*.log"
echo "  2. Backtest top-performing models on held-out data"
echo "  3. Deploy best models to paper trading"
echo ""
echo "================================================================================"
