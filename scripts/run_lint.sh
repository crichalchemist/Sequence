#!/bin/bash
# Run code quality checks (linting and formatting)
# Usage: ./scripts/run_lint.sh [--fix] [--check-only]

set -e  # Exit on error

# Parse arguments
FIX_MODE=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            FIX_MODE=true
            shift
            ;;
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--fix] [--check-only]"
            exit 1
            ;;
    esac
done

echo "=================================="
echo "  🔍 Running Code Quality Checks"
echo "=================================="

# Activate virtual environment if exists
if [ -d ".venvx" ]; then
    source .venvx/bin/activate
fi

# Check if ruff is installed
if ! python -c "import ruff" 2>/dev/null; then
    echo "⚠️  Ruff not found. Installing..."
    pip install ruff --quiet
fi

EXIT_CODE=0

# Run ruff linting
echo ""
echo "Running ruff linter..."
if [ "$FIX_MODE" = true ]; then
    python -m ruff check . --fix || EXIT_CODE=$?
else
    python -m ruff check . || EXIT_CODE=$?
fi

# Run ruff formatting check (if not fixing)
if [ "$CHECK_ONLY" = true ]; then
    echo ""
    echo "Checking code formatting..."
    python -m ruff format --check . || EXIT_CODE=$?
elif [ "$FIX_MODE" = true ]; then
    echo ""
    echo "Applying code formatting..."
    python -m ruff format . || EXIT_CODE=$?
fi

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All code quality checks passed!"
else
    echo "❌ Code quality issues found (exit code: $EXIT_CODE)"
    if [ "$FIX_MODE" = false ]; then
        echo ""
        echo "💡 Run with --fix to automatically fix issues:"
        echo "   ./scripts/run_lint.sh --fix"
    fi
fi

exit $EXIT_CODE
