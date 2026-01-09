#!/bin/bash

# Installation script for new fundamental data sources
# Installs UN Comtrade API and FRED packages from local directories

set -e  # Exit on error

echo "========================================"
echo "Installing Fundamental Data Sources"
echo "========================================"
echo ""

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Project root: $PROJECT_ROOT"
echo ""

# Check if the data collection directory exists
DATA_DIR="$PROJECT_ROOT/new data for collection"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Directory '$DATA_DIR' not found"
    echo "   Make sure the 'new data for collection' directory exists"
    exit 1
fi

# Install UN Comtrade API
echo "1. Installing UN Comtrade API..."
COMTRADE_DIR="$DATA_DIR/comtradeapicall"
if [ -d "$COMTRADE_DIR" ]; then
    pip install -e "$COMTRADE_DIR"
    echo "   ✅ UN Comtrade API installed"
else
    echo "   ⚠️  Warning: $COMTRADE_DIR not found, skipping"
fi
echo ""

# Install FRED (Federal Reserve Economic Data)
echo "2. Installing FRED API..."
FRB_DIR="$DATA_DIR/FRB"
if [ -d "$FRB_DIR" ]; then
    pip install -e "$FRB_DIR"
    echo "   ✅ FRED API installed"
else
    echo "   ⚠️  Warning: $FRB_DIR not found, skipping"
fi
echo ""

# ECB Shocks - no installation needed (CSV data only)
echo "3. Checking ECB Monetary Policy Shocks data..."
ECB_DIR="$DATA_DIR/jkshocks_update_ecb"
if [ -d "$ECB_DIR" ]; then
    if [ -f "$ECB_DIR/shocks_ecb_mpd_me_d.csv" ] && [ -f "$ECB_DIR/shocks_ecb_mpd_me_m.csv" ]; then
        echo "   ✅ ECB shocks data found (CSV files, no installation needed)"
    else
        echo "   ⚠️  Warning: ECB shock CSV files not found in $ECB_DIR"
    fi
else
    echo "   ⚠️  Warning: $ECB_DIR not found"
fi
echo ""

echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Set up API keys:"
echo "   export FRED_API_KEY='your_fred_key_here'"
echo "   export COMTRADE_API_KEY='your_comtrade_key_here'  # optional"
echo ""
echo "2. Get API keys:"
echo "   - FRED: https://fred.stlouisfed.org/docs/api/api_key.html (free)"
echo "   - Comtrade: https://comtradeplus.un.org/ (free tier available)"
echo ""
echo "3. Test the installation:"
echo "   python run/scripts/test_data_sources.py"
echo ""
echo "4. See documentation:"
echo "   - Quick start: docs/QUICK_START_DATA_SOURCES.md"
echo "   - Full docs: docs/NEW_DATA_SOURCES.md"
echo ""
