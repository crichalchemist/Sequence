# Quick Start: New Data Sources

## Installation (5 minutes)

```bash
# 1. Run installation script
bash run/scripts/install_data_sources.sh

# 2. Set up API keys
export FRED_API_KEY="your_fred_key_here"
export COMTRADE_API_KEY="your_comtrade_key_here"  # optional

# 3. Test installation
python run/scripts/test_data_sources.py
```

## Get API Keys

- **FRED**: Free at https://fred.stlouisfed.org/docs/api/api_key.html
- **Comtrade**: Free tier at https://comtradeplus.un.org/ (500 records/request)

## Collect Data (1 command)

```python
from data.extended_data_collection import collect_all_forex_fundamentals
import os

data = collect_all_forex_fundamentals(
    currency_pair="EURUSD",
    start_date="2023-01-01",
    end_date="2023-12-31",
    fred_api_key=os.getenv("FRED_API_KEY")
)

print(f"Collected {len(data['economic'])} economic indicators")
print(f"Collected {len(data['shocks'])} ECB shocks")
```

## Command Line Example

```bash
python run/scripts/example_fundamental_integration.py \
    --pair EURUSD \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --output-dir data/fundamentals \
    --price-data data/prepared/EURUSD_1h.parquet
```

## What You Get

### Trade Data (UN Comtrade)
- Monthly trade balance by country
- Tracks import/export flows
- Relevant for: EURUSD, GBPUSD, USDJPY, AUDUSD, etc.

### Economic Indicators (FRED)
- Interest rates (Fed Funds, ECB Deposit Rate, etc.)
- Inflation (CPI)
- GDP growth
- Unemployment rates

### Monetary Shocks (ECB)
- Policy surprise measures
- Hawkish/dovish signals
- EUR pairs only

## Files Created

```
data/fundamentals/
  ├── EURUSD_trade.parquet      # Trade balance data
  ├── EURUSD_economic.parquet   # Economic indicators
  ├── EURUSD_shocks.parquet     # ECB monetary shocks
  └── EURUSD_merged.parquet     # All data + prices
```

## Use in Training

```python
# Load merged dataset with fundamentals
import pandas as pd

df = pd.read_parquet("data/fundamentals/EURUSD_merged.parquet")

# Check fundamental features
fundamental_cols = [col for col in df.columns 
                   if col.startswith(('trade_', 'economic_', 'shocks_'))]

print(f"Added {len(fundamental_cols)} fundamental features")
# Example features:
# - economic_interest_rate
# - economic_inflation
# - trade_trade_balance
# - shocks_MP_median
```

## Troubleshooting

**Import error?**
```bash
pip install -e "new data for collection/comtradeapicall"
pip install -e "new data for collection/FRB"
```

**No data returned?**
- Check API keys are set
- FRED requires API key (free)
- Comtrade works without key (limited data)

**ECB shocks file not found?**
```bash
ls "new data for collection/jkshocks_update_ecb/"
# Should see .csv files
```

## Full Documentation

See [docs/NEW_DATA_SOURCES.md](../docs/NEW_DATA_SOURCES.md) for complete documentation.
