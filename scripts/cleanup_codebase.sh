#!/bin/bash
# Comprehensive codebase cleanup script
# Runs linting with auto-fixes and reports remaining issues
# Usage: ./scripts/cleanup_codebase.sh [--dry-run] [--aggressive]

set -e

DRY_RUN=false
AGGRESSIVE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --aggressive)
            AGGRESSIVE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--aggressive]"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "  🧹 Codebase Cleanup"
echo "========================================="

# Activate venv
if [ -d ".venvx" ]; then
    source .venvx/bin/activate
fi

# Ensure ruff is installed
if ! python -c "import ruff" 2>/dev/null; then
    echo "Installing ruff..."
    pip install ruff --quiet
fi

# Step 1: Show current state
echo ""
echo "📊 Current state:"
python -m ruff check . --statistics 2>&1 | tail -n 5

# Step 2: Safe auto-fixes
echo ""
echo "🔧 Step 1: Applying safe auto-fixes..."
if [ "$DRY_RUN" = true ]; then
    echo "   (Dry run - no changes will be made)"
    python -m ruff check . --diff | head -n 100
else
    python -m ruff check . --fix --show-fixes
    echo "   ✓ Safe fixes applied"
fi

# Step 3: Unsafe fixes (if aggressive mode)
if [ "$AGGRESSIVE" = true ]; then
    echo ""
    echo "🔧 Step 2: Applying unsafe fixes (aggressive mode)..."
    if [ "$DRY_RUN" = false ]; then
        python -m ruff check . --fix --unsafe-fixes --show-fixes
        echo "   ✓ Unsafe fixes applied"
    fi
fi

# Step 4: Format code
echo ""
echo "💅 Step 3: Formatting code..."
if [ "$DRY_RUN" = true ]; then
    python -m ruff format --check . || true
else
    python -m ruff format .
    echo "   ✓ Code formatted"
fi

# Step 5: Show remaining issues
echo ""
echo "📋 Remaining issues:"
python -m ruff check . --statistics 2>&1 | tail -n 20

# Step 6: Critical issues only
echo ""
echo "🚨 Critical issues (syntax errors, unused variables):"
python -m ruff check . --select E,F --no-fix | head -n 50 || true

echo ""
echo "========================================="
if [ "$DRY_RUN" = true ]; then
    echo "✓ Dry run complete (no changes made)"
else
    echo "✓ Cleanup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Review remaining issues above"
    echo "  2. Run tests: ./scripts/run_tests.sh"
    echo "  3. Commit changes: git add -A && git commit -m 'chore: lint and format codebase'"
fi
echo "========================================="
