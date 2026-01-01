#!/bin/bash
# Automatic Training Manager
# Continuously monitors training completion and starts next batch automatically
#
# Features:
# - Waits for current batch to complete
# - Prepares next available pairs
# - Starts training in parallel (2 at a time)
# - Loops until all pairs are trained

set -e

PROJECT_ROOT="/Volumes/Containers/Sequence"
cd "$PROJECT_ROOT"

LOG_FILE="$PROJECT_ROOT/training_logs/auto_train_manager_$(date +%Y%m%d_%H%M%S).log"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

echo ""
echo "================================================================================"
echo "  🤖 AUTOMATIC TRAINING MANAGER"
echo "================================================================================"
echo "  Strategy: Continuous parallel training (2 pairs at a time)"
echo "  Log: $LOG_FILE"
echo "================================================================================"
echo ""

# Get list of already trained pairs
get_trained_pairs() {
    find checkpoints -name "policy_epoch_100.pt" 2>/dev/null | cut -d'/' -f2 | sort -u
}

# Get list of pairs with CSV files ready
get_ready_pairs() {
    find data/histdata -name "*.csv" -type f 2>/dev/null | cut -d'/' -f3 | sort -u
}

# Get list of pairs with prepared .npy files
get_prepared_pairs() {
    find data/prepared -name "*.npy" -type f 2>/dev/null | cut -d'/' -f3 | sort -u
}

# Check if pair is currently training
is_training() {
    local pair=$1
    local pair_lower=$(echo "$pair" | tr '[:upper:]' '[:lower:]')

    # Check for recent log file (created in last 3 hours)
    local recent_log=$(find training_logs/$pair_lower -name "training_*.log" -mmin -180 2>/dev/null)

    if [ -n "$recent_log" ]; then
        # Check if training is complete
        if grep -q "TRAINING COMPLETE" "$recent_log" 2>/dev/null; then
            return 1  # Not training (complete)
        else
            return 0  # Still training
        fi
    fi

    return 1  # Not training
}

# Main loop
BATCH_NUM=1
MAX_BATCHES=20  # Safety limit

while [ $BATCH_NUM -le $MAX_BATCHES ]; do
    echo ""
    echo "================================================================================"
    log "BATCH $BATCH_NUM: Finding next pairs to train..."
    echo "================================================================================"

    # Get trained pairs
    TRAINED_PAIRS=($(get_trained_pairs))
    log "Already trained: ${#TRAINED_PAIRS[@]} pairs"

    # Get pairs with CSV files
    READY_PAIRS=($(get_ready_pairs))
    log "Pairs with CSV files: ${#READY_PAIRS[@]}"

    # Find pairs that are ready but not yet trained
    CANDIDATES=()
    for pair in "${READY_PAIRS[@]}"; do
        pair_upper=$(echo "$pair" | tr '[:lower:]' '[:upper:]')

        # Skip if already trained
        if printf '%s\n' "${TRAINED_PAIRS[@]}" | grep -q "^${pair}$"; then
            continue
        fi

        # Skip if currently training
        if is_training "$pair_upper"; then
            log "  $pair_upper is currently training, skipping..."
            continue
        fi

        CANDIDATES+=("$pair")
    done

    log "Candidate pairs for training: ${#CANDIDATES[@]}"

    if [ ${#CANDIDATES[@]} -eq 0 ]; then
        log_warning "No new pairs ready for training."
        log "Waiting 10 minutes for data collection to prepare more pairs..."
        sleep 600  # Wait 10 minutes
        continue
    fi

    # Select next 2 pairs to train
    NEXT_PAIRS=()
    for i in 0 1; do
        if [ $i -lt ${#CANDIDATES[@]} ]; then
            NEXT_PAIRS+=("${CANDIDATES[$i]}")
        fi
    done

    if [ ${#NEXT_PAIRS[@]} -eq 0 ]; then
        log_warning "No pairs available for this batch"
        break
    fi

    log "Selected for Batch $BATCH_NUM: ${NEXT_PAIRS[*]}"
    echo ""

    # Prepare and train each pair
    for pair in "${NEXT_PAIRS[@]}"; do
        pair_upper=$(echo "$pair" | tr '[:lower:]' '[:upper:]')

        # Check if already prepared
        if [ -d "data/prepared/$pair" ] && [ "$(ls -A data/prepared/$pair/*.npy 2>/dev/null | wc -l)" -gt 0 ]; then
            log_success "$pair_upper already prepared"
        else
            log "Preparing $pair_upper..."
            mkdir -p "data/prepared/$pair"

            if SEQ_LOG_LEVEL=INFO python3 scripts/validate_training_data.py \
                --pair "$pair_upper" \
                --data-root data/histdata \
                --prepare \
                --output "data/prepared/$pair" \
                --sequence-length 390 >> "$LOG_FILE" 2>&1; then

                npy_count=$(ls -1 data/prepared/$pair/*.npy 2>/dev/null | wc -l)
                log_success "$pair_upper prepared ($npy_count files)"
            else
                log_warning "$pair_upper preparation failed, skipping..."
                continue
            fi
        fi

        # Start training
        log "Starting training for $pair_upper..."
        bash scripts/run_per_pair_training.sh "$pair_upper" \
            --epochs 100 \
            --episodes-per-epoch 200 \
            --seed 42 >> "$LOG_FILE" 2>&1 &

        TRAIN_PID=$!
        log "  Training PID: $TRAIN_PID"
        sleep 2  # Brief delay between starts
    done

    # Wait for current batch to complete
    log "Waiting for Batch $BATCH_NUM to complete..."
    log "  Training: ${NEXT_PAIRS[*]}"

    WAIT_COUNT=0
    MAX_WAIT=180  # 3 hours max per batch

    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        ALL_COMPLETE=true

        for pair in "${NEXT_PAIRS[@]}"; do
            pair_upper=$(echo "$pair" | tr '[:lower:]' '[:upper:]')

            # Check if checkpoint exists
            if [ ! -f "checkpoints/$pair/production_*/policy_epoch_100.pt" ]; then
                ALL_COMPLETE=false
                break
            fi
        done

        if [ "$ALL_COMPLETE" = true ]; then
            break
        fi

        # Wait 1 minute
        sleep 60
        WAIT_COUNT=$((WAIT_COUNT + 1))

        # Log progress every 15 minutes
        if [ $((WAIT_COUNT % 15)) -eq 0 ]; then
            HOURS=$((WAIT_COUNT / 60))
            MINS=$((WAIT_COUNT % 60))
            log "Batch $BATCH_NUM still training... (${HOURS}h ${MINS}m elapsed)"
        fi
    done

    if [ "$ALL_COMPLETE" = true ]; then
        log_success "Batch $BATCH_NUM complete!"

        # Show completion status
        for pair in "${NEXT_PAIRS[@]}"; do
            if [ -f "checkpoints/$pair/production_*/policy_epoch_100.pt" ]; then
                log_success "  ✅ $pair"
            fi
        done
    else
        log_warning "Batch $BATCH_NUM timed out after ${MAX_WAIT} minutes"
    fi

    BATCH_NUM=$((BATCH_NUM + 1))
    log "Moving to next batch..."
    sleep 10
done

echo ""
echo "================================================================================"
echo "  ✅ AUTO TRAINING MANAGER COMPLETE"
echo "================================================================================"
log_success "Training manager finished after $BATCH_NUM batches"
echo ""
echo "Check trained models: ls -d checkpoints/*/production_*/"
echo "Training log: $LOG_FILE"
echo ""
