#!/bin/bash
# Overnight Auto-Training Queue
# Waits for EURUSD to complete, then trains remaining major pairs sequentially
#
# Usage: bash train_overnight_queue.sh &

set -e

PROJECT_ROOT=$(pwd)
LOG_FILE="$PROJECT_ROOT/training_logs/overnight_queue_$(date +%Y%m%d_%H%M%S).log"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_FILE"
}

echo ""
echo "================================================================================"
echo "  OVERNIGHT TRAINING QUEUE"
echo "================================================================================"
echo "  Strategy: Wait for EURUSD → Train 5 major pairs sequentially"
echo "  Queue: EURGBP, EURJPY, GBPUSD, USDJPY, AUDUSD"
echo "  Log: $LOG_FILE"
echo "================================================================================"
echo ""

#------------------------------------------------------------------------------
# STEP 1: WAIT FOR EURUSD TO COMPLETE
#------------------------------------------------------------------------------
log "Waiting for EURUSD training to complete..."
log "Checking for EURUSD checkpoint: policy_epoch_100.pt"

EURUSD_CHECKPOINT=""
WAIT_COUNT=0
MAX_WAIT=720  # Max 6 hours (720 minutes)

while [ -z "$EURUSD_CHECKPOINT" ] && [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    # Check if EURUSD epoch 100 checkpoint exists
    EURUSD_CHECKPOINT=$(find checkpoints/eurusd/production_*/policy_epoch_100.pt 2>/dev/null | head -1)

    if [ -z "$EURUSD_CHECKPOINT" ]; then
        # Not done yet, wait 1 minute
        sleep 60
        WAIT_COUNT=$((WAIT_COUNT + 1))

        # Log progress every 15 minutes
        if [ $((WAIT_COUNT % 15)) -eq 0 ]; then
            HOURS=$((WAIT_COUNT / 60))
            MINS=$((WAIT_COUNT % 60))
            log "Still waiting for EURUSD... (${HOURS}h ${MINS}m elapsed)"

            # Check current epoch
            CURRENT_EPOCH=$(grep -c "^Epoch.*Summary" training_logs/eurusd/training_*.log 2>/dev/null || echo "0")
            log "  EURUSD progress: ~${CURRENT_EPOCH}/100 epochs"
        fi
    fi
done

if [ -z "$EURUSD_CHECKPOINT" ]; then
    log "ERROR: EURUSD training did not complete within ${MAX_WAIT} minutes"
    log "Queue aborted. Please check EURUSD training status."
    exit 1
fi

log_success "EURUSD training complete! Found checkpoint: $EURUSD_CHECKPOINT"
echo ""

# Brief pause to let system settle
sleep 10

#------------------------------------------------------------------------------
# STEP 2: TRAIN MAJOR PAIRS SEQUENTIALLY
#------------------------------------------------------------------------------
log "Starting sequential training of major pairs..."
echo ""

# Define training queue (pairs most likely to have data ready)
PAIRS=(
    "EURGBP"   # EUR/GBP - major cross
    "EURJPY"   # EUR/JPY - major cross
    "GBPUSD"   # GBP/USD - cable
    "USDJPY"   # USD/JPY - major
    "AUDUSD"   # AUD/USD - commodity currency
)

COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

for i in "${!PAIRS[@]}"; do
    PAIR="${PAIRS[$i]}"
    PAIR_NUM=$((i + 1))
    TOTAL_PAIRS=${#PAIRS[@]}

    echo ""
    echo "================================================================================"
    log "[$PAIR_NUM/$TOTAL_PAIRS] Starting training for $PAIR"
    echo "================================================================================"

    # Check if data exists
    PAIR_LOWER=$(echo "$PAIR" | tr '[:upper:]' '[:lower:]')
    DATA_DIR="data/prepared/$PAIR_LOWER"

    if [ ! -d "$DATA_DIR" ]; then
        log "⚠️  WARNING: No prepared data found for $PAIR at $DATA_DIR"
        log "Skipping $PAIR and continuing to next pair..."
        FAILED=$((FAILED + 1))
        continue
    fi

    NPY_COUNT=$(ls -1 "$DATA_DIR"/*.npy 2>/dev/null | wc -l | tr -d ' ')
    if [ "$NPY_COUNT" -eq 0 ]; then
        log "⚠️  WARNING: No .npy files found for $PAIR"
        log "Skipping $PAIR and continuing to next pair..."
        FAILED=$((FAILED + 1))
        continue
    fi

    log "Found $NPY_COUNT prepared files for $PAIR"

    # Train this pair
    PAIR_START=$(date +%s)

    if bash scripts/run_per_pair_training.sh "$PAIR" \
        --epochs 100 \
        --episodes-per-epoch 200 \
        --seed 42; then

        PAIR_END=$(date +%s)
        PAIR_DURATION=$((PAIR_END - PAIR_START))
        PAIR_HOURS=$((PAIR_DURATION / 3600))
        PAIR_MINS=$(((PAIR_DURATION % 3600) / 60))

        log_success "$PAIR training complete in ${PAIR_HOURS}h ${PAIR_MINS}m"
        COMPLETED=$((COMPLETED + 1))

        # Show latest checkpoint
        LATEST_CHECKPOINT=$(ls -t checkpoints/$PAIR_LOWER/production_*/policy_epoch_100.pt 2>/dev/null | head -1)
        if [ -n "$LATEST_CHECKPOINT" ]; then
            log "  Checkpoint: $LATEST_CHECKPOINT"
        fi
    else
        log "❌ ERROR: Training failed for $PAIR"
        FAILED=$((FAILED + 1))
        log "Continuing to next pair..."
    fi

    # Progress update
    REMAINING=$((TOTAL_PAIRS - PAIR_NUM))
    if [ $REMAINING -gt 0 ]; then
        log "Progress: $PAIR_NUM/$TOTAL_PAIRS complete, $REMAINING remaining"
        echo ""
    fi
done

#------------------------------------------------------------------------------
# COMPLETION SUMMARY
#------------------------------------------------------------------------------
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINS=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "================================================================================"
echo "  ✅ OVERNIGHT QUEUE COMPLETE"
echo "================================================================================"
echo ""
log_success "Training queue finished!"
echo ""
echo "Results:"
echo "  - Pairs trained successfully: $COMPLETED"
echo "  - Pairs failed/skipped: $FAILED"
echo "  - Total duration: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
echo "  - Models saved to: checkpoints/{pair}/production_*/"
echo "  - Queue log: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Review individual training logs: training_logs/{pair}/training_*.log"
echo "  2. Check model performance and select best checkpoints"
echo "  3. Queue remaining pairs or wait for GPU cluster"
echo ""
echo "Trained pairs (including EURUSD): $((COMPLETED + 1))"
echo "Remaining pairs: $((35 - COMPLETED - 1))"
echo ""
echo "================================================================================"
