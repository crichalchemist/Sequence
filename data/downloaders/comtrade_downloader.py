"""
UN Comtrade International Trade Data Downloader

Downloads international trade statistics to analyze cross-border trade flows
that may impact forex markets. Trade balance data is particularly relevant
for currency valuation and economic sentiment.

Data Source:
- UN Comtrade API: https://comtrade.un.org
- Trade flows, imports/exports, trade balance by country/commodity

Dependencies:
    Install from local package:
    pip install -e "new data for collection/comtradeapicall"

Usage:
    from data.downloaders.comtrade_downloader import (
        download_trade_balance,
        download_bilateral_trade,
        get_trade_indicators_for_forex
    )

    # Get trade balance for EUR/USD relevant countries
    trade_data = get_trade_indicators_for_forex(
        currency_pair="EURUSD",
        start_date="2023-01-01",
        end_date="2023-12-31",
        subscription_key=os.getenv("COMTRADE_API_KEY")
    )
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)

# Country mapping for major forex pairs
FOREX_COUNTRY_MAPPING = {
    "EURUSD": {"base": ["276"], "quote": ["842"]},  # Germany (EU proxy), USA
    "GBPUSD": {"base": ["826"], "quote": ["842"]},  # UK, USA
    "USDJPY": {"base": ["842"], "quote": ["392"]},  # USA, Japan
    "AUDUSD": {"base": ["036"], "quote": ["842"]},  # Australia, USA
    "EURJPY": {"base": ["276"], "quote": ["392"]},  # Germany (EU proxy), Japan
    "EURGBP": {"base": ["276"], "quote": ["826"]},  # Germany (EU proxy), UK
    "USDCAD": {"base": ["842"], "quote": ["124"]},  # USA, Canada
    "USDCHF": {"base": ["842"], "quote": ["756"]},  # USA, Switzerland
}


def download_trade_balance(
        reporter_code: str,
        start_year: int,
        end_year: int,
        subscription_key: str | None = None
) -> pd.DataFrame:
    """
    Download trade balance data for a country.

    Args:
        reporter_code: UN Comtrade country code (e.g., "842" for USA)
        start_year: Start year (e.g., 2020)
        end_year: End year (e.g., 2023)
        subscription_key: UN Comtrade API subscription key (required for bulk downloads)

    Returns:
        DataFrame with columns: [date, trade_balance, imports, exports, country_code]

    Example:
        >>> df = download_trade_balance("842", 2023, 2023, subscription_key)
        >>> print(df.head())
                date  trade_balance  imports    exports country_code
        0 2023-01-01      -50000000  150000000  100000000         842
    """
    try:
        from comtradeapicall import previewGet
    except ImportError:
        raise ImportError(
            "comtradeapicall not installed. Install with: "
            "pip install -e 'new data for collection/comtradeapicall'"
        )

    logger.info(f"[comtrade] Downloading trade balance for country {reporter_code}, "
                f"{start_year}-{end_year}")

    try:
        # Build selection criteria
        selection_criteria = {
            "typeCode": "C",  # Commodities
            "freqCode": "M",  # Monthly
            "clCode": "HS",  # Harmonized System
            "period": [str(year) for year in range(start_year, end_year + 1)],
            "reporterCode": reporter_code,
            "cmdCode": "TOTAL",  # All commodities
            "flowCode": ["M", "X"],  # Imports and Exports
            "partnerCode": None,  # All partners
            "partner2Code": None,
        }

        # Use preview if no subscription key, otherwise use full API
        if subscription_key:
            df = previewGet.getTradeBalance(
                subscription_key=subscription_key,
                **selection_criteria
            )
        else:
            logger.warning("[comtrade] No subscription key provided, using preview (limited to 500 records)")
            df = previewGet.previewTradeBalance(**selection_criteria)

        if df is None or df.empty:
            logger.warning(f"[comtrade] No data returned for country {reporter_code}")
            return pd.DataFrame()

        # Process and normalize
        df = df.copy()
        df['date'] = pd.to_datetime(df['period'], format='%Y%m')
        df['country_code'] = reporter_code

        # Calculate trade balance if not already present
        if 'trade_balance' not in df.columns:
            df['trade_balance'] = df.get('exports', 0) - df.get('imports', 0)

        # Select and rename columns
        result = df[['date', 'trade_balance', 'country_code']].copy()

        logger.info(f"[comtrade] Downloaded {len(result)} trade balance records")
        return result

    except Exception as e:
        logger.error(f"[comtrade] Error downloading trade balance: {e}")
        raise


def get_trade_indicators_for_forex(
        currency_pair: str,
        start_year: int,
        end_year: int,
        subscription_key: str | None = None
) -> pd.DataFrame:
    """
    Download trade indicators relevant to a forex pair.

    Args:
        currency_pair: Forex pair symbol (e.g., "EURUSD", "GBPUSD")
        start_year: Start year
        end_year: End year
        subscription_key: UN Comtrade API subscription key

    Returns:
        DataFrame with trade balance data for both countries in the pair

    Example:
        >>> df = get_trade_indicators_for_forex("EURUSD", 2023, 2023)
        >>> print(df.head())
                date currency_pair country_type  trade_balance country_code
        0 2023-01-01       EURUSD         base       15000000          276
        1 2023-02-01       EURUSD         base       12000000          276
    """
    pair_upper = currency_pair.upper()

    if pair_upper not in FOREX_COUNTRY_MAPPING:
        raise ValueError(
            f"Currency pair {currency_pair} not supported. "
            f"Supported pairs: {list(FOREX_COUNTRY_MAPPING.keys())}"
        )

    countries = FOREX_COUNTRY_MAPPING[pair_upper]
    all_data = []

    # Download for base currency country
    for base_country in countries["base"]:
        df_base = download_trade_balance(
            base_country, start_year, end_year, subscription_key
        )
        if not df_base.empty:
            df_base['currency_pair'] = pair_upper
            df_base['country_type'] = 'base'
            all_data.append(df_base)

    # Download for quote currency country
    for quote_country in countries["quote"]:
        df_quote = download_trade_balance(
            quote_country, start_year, end_year, subscription_key
        )
        if not df_quote.empty:
            df_quote['currency_pair'] = pair_upper
            df_quote['country_type'] = 'quote'
            all_data.append(df_quote)

    if not all_data:
        logger.warning(f"[comtrade] No data retrieved for {currency_pair}")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values('date').reset_index(drop=True)

    logger.info(f"[comtrade] Retrieved {len(result)} trade indicator records for {currency_pair}")
    return result


def format_for_cognee(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format Comtrade data for Cognee ingestion.

    Args:
        df: Raw Comtrade DataFrame

    Returns:
        DataFrame with 'full_text' column for semantic search
    """
    df = df.copy()

    df['full_text'] = df.apply(
        lambda row: (
            f"Trade balance for {row.get('country_type', 'country')} currency "
            f"in {row.get('currency_pair', 'pair')} on {row['date'].strftime('%Y-%m-%d')}: "
            f"${row['trade_balance']:,.0f}. "
            f"Country code: {row.get('country_code', 'N/A')}."
        ),
        axis=1
    )

    return df
