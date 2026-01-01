#!/bin/bash
# Multi-Pair FX/Crypto Data Collection Pipeline
# Downloads and prepares data for ALL currency pairs in pairs.csv
#
# Usage:
#   bash scripts/run_data_collection.sh
#   bash scripts/run_data_collection.sh --start-year 2014 --end-year 2024

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
START_YEAR=2014
END_YEAR=2024
MAX_DOWNLOADS=1000  # Increased for multi-pair collection
SEQUENCE_LENGTH=390
PAIRS_CSV="$PROJECT_ROOT/pairs.csv"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-year)
            START_YEAR="$2"
            shift 2
            ;;
        --end-year)
            END_YEAR="$2"
            shift 2
            ;;
        --max-downloads)
            MAX_DOWNLOADS="$2"
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
LOG_FILE="$LOG_DIR/data_collection_${TIMESTAMP}.log"

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
echo "  MULTI-PAIR DATA COLLECTION PIPELINE"
echo "================================================================================"
echo "  Years: $START_YEAR - $END_YEAR"
echo "  Pairs CSV: $PAIRS_CSV"
echo "  Max downloads: $MAX_DOWNLOADS"
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
log "Found $TOTAL_PAIRS currency pairs in CSV"

#------------------------------------------------------------------------------
# PHASE 1: DOWNLOAD ALL PAIRS
#------------------------------------------------------------------------------
log "PHASE 1/3: Downloading historical data for all pairs..."
log "Time range: $START_YEAR to $END_YEAR"

if SEQ_LOG_LEVEL=INFO python3 data/download_all_fx_data.py \
    --start-year $START_YEAR \
    --end-year $END_YEAR \
    --max-downloads $MAX_DOWNLOADS \
    --output data/histdata >> "$LOG_FILE" 2>&1; then
    log_success "Data download complete for all pairs"
else
    log_error "Data download failed. Check log: $LOG_FILE"
    exit 1
fi

#------------------------------------------------------------------------------
# PHASE 2: EXTRACT ALL ZIPS
#------------------------------------------------------------------------------
log "PHASE 2/3: Extracting ZIP files for all pairs..."

PAIRS_PROCESSED=0
PAIRS_FAILED=0

# Read pairs from CSV and extract each
tail -n +2 "$PAIRS_CSV" | while IFS=',' read -r currency_pair_name pair history_first_trading_month; do
    PAIR_LOWER=$(echo "$pair" | tr '[:upper:]' '[:lower:]')
    DATA_DIR="data/histdata/$PAIR_LOWER"

    if [ ! -d "$DATA_DIR" ]; then
        log_warning "Data directory not found for $pair: $DATA_DIR (skipping)"
        continue
    fi

    log "Extracting ZIPs for $pair..."

    cd "$DATA_DIR"
    if unzip -o "*.zip" >> "$LOG_FILE" 2>&1; then
        log_success "Extracted ZIPs for $pair"
    else
        log_warning "Some ZIPs failed for $pair (OK if CSVs exist)"
    fi
    cd "$PROJECT_ROOT"

    # Count CSV files
    CSV_COUNT=$(ls -1 "$DATA_DIR"/*.csv 2>/dev/null | wc -l | tr -d ' ')
    log "  Found $CSV_COUNT CSV files for $pair"

    if [ "$CSV_COUNT" -eq 0 ]; then
        log_error "No CSV files found for $pair after extraction"
        PAIRS_FAILED=$((PAIRS_FAILED + 1))
    else
        PAIRS_PROCESSED=$((PAIRS_PROCESSED + 1))
    fi
done

log "Extraction complete. Processed: $PAIRS_PROCESSED pairs"

#------------------------------------------------------------------------------
# PHASE 3: PREPARE DATA FOR ALL PAIRS
#------------------------------------------------------------------------------
log "PHASE 3/3: Preparing data for training (all pairs)..."
log "Sequence length: $SEQUENCE_LENGTH steps"

PAIRS_PREPARED=0
PAIRS_PREP_FAILED=0

# Read pairs from CSV and prepare each
tail -n +2 "$PAIRS_CSV" | while IFS=',' read -r currency_pair_name pair history_first_trading_month; do
    PAIR_LOWER=$(echo "$pair" | tr '[:upper:]' '[:lower:]')
    DATA_DIR="data/histdata/$PAIR_LOWER"

    if [ ! -d "$DATA_DIR" ]; then
        log_warning "Skipping $pair - no data directory"
        continue
    fi

    # Count CSV files to verify we have data
    CSV_COUNT=$(ls -1 "$DATA_DIR"/*.csv 2>/dev/null | wc -l | tr -d ' ')
    if [ "$CSV_COUNT" -eq 0 ]; then
        log_warning "Skipping $pair - no CSV files found"
        continue
    fi

    log "Preparing data for $pair ($CSV_COUNT CSV files)..."

    # Create pair-specific output directory
    OUTPUT_DIR="data/prepared/$PAIR_LOWER"
    mkdir -p "$OUTPUT_DIR"

    if SEQ_LOG_LEVEL=INFO python3 scripts/validate_training_data.py \
        --pair "$pair" \
        --data-root data \
        --prepare \
        --output "$OUTPUT_DIR" \
        --sequence-length $SEQUENCE_LENGTH >> "$LOG_FILE" 2>&1; then

        # Count prepared files
        NPY_COUNT=$(ls -1 "$OUTPUT_DIR"/*.npy 2>/dev/null | wc -l | tr -d ' ')
        log_success "Prepared $NPY_COUNT .npy files for $pair"
        PAIRS_PREPARED=$((PAIRS_PREPARED + 1))
    else
        log_error "Data preparation failed for $pair"
        PAIRS_PREP_FAILED=$((PAIRS_PREP_FAILED + 1))
    fi
done

#------------------------------------------------------------------------------
# COMPLETION SUMMARY
#------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "  ✅ DATA COLLECTION COMPLETE"
echo "================================================================================"
echo ""
log_success "All phases completed!"
echo ""
echo "Results:"
echo "  - Total pairs in CSV: $TOTAL_PAIRS"
echo "  - Pairs with extracted data: $PAIRS_PROCESSED"
echo "  - Pairs with prepared data: $PAIRS_PREPARED"
echo "  - Preparation failures: $PAIRS_PREP_FAILED"
echo "  - Prepared data location: data/prepared/{pair}/*.npy"
echo "  - Collection log: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Review log for any failures: $LOG_FILE"
echo "  2. Train models per pair:"
echo "     bash scripts/run_per_pair_training.sh <PAIR>"
echo "  3. Or train all pairs:"
echo "     bash scripts/run_all_training.sh"
echo ""
echo "================================================================================"
