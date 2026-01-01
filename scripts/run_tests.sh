#!/bin/bash
# Run test suite with coverage reporting
# Usage: ./scripts/run_tests.sh [--fast] [--verbose]

set -e  # Exit on error

# Parse arguments
FAST_MODE=false
VERBOSE=""
TEST_PATTERN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            FAST_MODE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        --pattern|-k)
            TEST_PATTERN="-k $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--fast] [--verbose] [--pattern <test_pattern>]"
            exit 1
            ;;
    esac
done

echo "=================================="
echo "  🧪 Running Test Suite"
echo "=================================="

# Activate virtual environment if exists
if [ -d ".venvx" ]; then
    source .venvx/bin/activate
fi

# Run tests based on mode
if [ "$FAST_MODE" = true ]; then
    echo "Running fast tests (smoke tests only)..."
    python -m pytest tests/test_smoke.py $VERBOSE $TEST_PATTERN
else
    echo "Running full test suite..."
    python -m pytest tests/ \
        $VERBOSE \
        $TEST_PATTERN \
        --tb=short \
        --disable-warnings \
        -x  # Stop on first failure
fi

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
else
    echo ""
    echo "❌ Tests failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
