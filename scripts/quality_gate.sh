#!/bin/bash
# Quality gate: Run all code quality checks before committing
# This should pass before pushing to main or creating PRs
# Usage: ./scripts/quality_gate.sh [--fast]

set -e

FAST_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            FAST_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--fast]"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "  🚦 Quality Gate"
echo "========================================="

EXIT_CODE=0

# Activate venv
if [ -d ".venvx" ]; then
    source .venvx/bin/activate
fi

# Step 1: Linting
echo ""
echo "Step 1/2: Running linter..."
echo "---"
if python -m ruff check . --quiet; then
    echo "✅ Lint check passed"
else
    echo "❌ Lint check failed"
    EXIT_CODE=1
    echo ""
    echo "Run this to see issues:"
    echo "  ./scripts/run_lint.sh"
    echo ""
    echo "Run this to auto-fix:"
    echo "  ./scripts/run_lint.sh --fix"
fi

# Step 2: Tests
echo ""
echo "Step 2/2: Running tests..."
echo "---"
if [ "$FAST_MODE" = true ]; then
    TEST_CMD="./scripts/run_tests.sh --fast"
else
    TEST_CMD="./scripts/run_tests.sh"
fi

if $TEST_CMD >/dev/null 2>&1; then
    echo "✅ Tests passed"
else
    echo "❌ Tests failed"
    EXIT_CODE=1
    echo ""
    echo "Run this to see details:"
    echo "  $TEST_CMD"
fi

# Summary
echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Quality gate PASSED"
    echo ""
    echo "Safe to commit and push!"
else
    echo "❌ Quality gate FAILED"
    echo ""
    echo "Please fix issues before committing."
fi
echo "========================================="

exit $EXIT_CODE
