"""
ECB Monetary Policy Shocks Downloader

Loads and processes ECB monetary policy and central bank information shocks
from Jarocinski & Karadi (2020). These shocks are valuable for forex trading
as they capture surprise changes in monetary policy sentiment.

Data Source:
- Jarocinski, M. and Karadi, P. (2020)
  "Deconstructing Monetary Policy Surprises - The Role of Information Shocks"
  AEJ:Macro, https://doi.org/10.1257/mac.20180090
- Updated ECB monetary policy shocks dataset

Dependencies:
    No external dependencies required - data is in CSV format

Usage:
    from data.downloaders.ecb_shocks_downloader import (
        load_ecb_shocks_daily,
        load_ecb_shocks_monthly,
        get_monetary_policy_events
    )

    # Load daily shocks
    daily_shocks = load_ecb_shocks_daily()

    # Load monthly aggregated shocks
    monthly_shocks = load_ecb_shocks_monthly()
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)

# Path to ECB shocks data
ECB_SHOCKS_DIR = ROOT / "new data for collection" / "jkshocks_update_ecb"
DAILY_SHOCKS_FILE = ECB_SHOCKS_DIR / "shocks_ecb_mpd_me_d.csv"
MONTHLY_SHOCKS_FILE = ECB_SHOCKS_DIR / "shocks_ecb_mpd_me_m.csv"


def load_ecb_shocks_daily() -> pd.DataFrame:
    """
    Load daily ECB monetary policy shocks (by Governing Council meeting).

    Returns:
        DataFrame with columns:
        - date: Meeting date
        - pc1: Surprise in policy indicator (1st principal component of OIS changes)
        - STOXX50: Euro Stoxx 50 changes during event window
        - MP_pm: Monetary Policy shock (Poor Man's sign restrictions)
        - CBI_pm: Central Bank Information shock (Poor Man's sign restrictions)
        - MP_median: Monetary Policy shock (median rotation)
        - CBI_median: Central Bank Information shock (median rotation)

    Example:
        >>> df = load_ecb_shocks_daily()
        >>> print(df.head())
                date      pc1  STOXX50   MP_pm  CBI_pm  MP_median  CBI_median
        0 1999-01-04  0.12345  0.23456  0.1000  0.0234     0.0950      0.0284
    """
    if not DAILY_SHOCKS_FILE.exists():
        raise FileNotFoundError(
            f"ECB daily shocks file not found: {DAILY_SHOCKS_FILE}. "
            "Ensure 'new data for collection/jkshocks_update_ecb/shocks_ecb_mpd_me_d.csv' exists."
        )

    logger.info(f"[ecb_shocks] Loading daily shocks from {DAILY_SHOCKS_FILE}")

    try:
        df = pd.read_csv(DAILY_SHOCKS_FILE)

        # Parse date column (first column)
        date_col = df.columns[0]
        df['date'] = pd.to_datetime(df[date_col])
        df = df.drop(columns=[date_col])

        # Ensure numeric columns
        numeric_cols = ['pc1', 'STOXX50', 'MP_pm', 'CBI_pm', 'MP_median', 'CBI_median']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"[ecb_shocks] Loaded {len(df)} daily shock observations")
        return df

    except Exception as e:
        logger.error(f"[ecb_shocks] Error loading daily shocks: {e}")
        raise


def load_ecb_shocks_monthly() -> pd.DataFrame:
    """
    Load monthly aggregated ECB monetary policy shocks.
    Zero values indicate no shocks in that month.

    Returns:
        DataFrame with same columns as daily data, aggregated to monthly frequency

    Example:
        >>> df = load_ecb_shocks_monthly()
        >>> print(df.head())
                date      pc1  STOXX50   MP_pm  CBI_pm  MP_median  CBI_median
        0 1999-01-01  0.12345  0.23456  0.1000  0.0234     0.0950      0.0284
    """
    if not MONTHLY_SHOCKS_FILE.exists():
        raise FileNotFoundError(
            f"ECB monthly shocks file not found: {MONTHLY_SHOCKS_FILE}. "
            "Ensure 'new data for collection/jkshocks_update_ecb/shocks_ecb_mpd_me_m.csv' exists."
        )

    logger.info(f"[ecb_shocks] Loading monthly shocks from {MONTHLY_SHOCKS_FILE}")

    try:
        df = pd.read_csv(MONTHLY_SHOCKS_FILE)

        # Parse date column (first column)
        date_col = df.columns[0]
        df['date'] = pd.to_datetime(df[date_col])
        df = df.drop(columns=[date_col])

        # Ensure numeric columns
        numeric_cols = ['pc1', 'STOXX50', 'MP_pm', 'CBI_pm', 'MP_median', 'CBI_median']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"[ecb_shocks] Loaded {len(df)} monthly shock observations")
        return df

    except Exception as e:
        logger.error(f"[ecb_shocks] Error loading monthly shocks: {e}")
        raise


def get_monetary_policy_events(
        start_date: str | None = None,
        end_date: str | None = None,
        frequency: str = "daily",
        shock_threshold: float | None = None
) -> pd.DataFrame:
    """
    Get ECB monetary policy events within a date range, optionally filtered by shock magnitude.

    Args:
        start_date: Start date in "YYYY-MM-DD" format (optional)
        end_date: End date in "YYYY-MM-DD" format (optional)
        frequency: "daily" or "monthly"
        shock_threshold: Optional minimum absolute shock magnitude to include
                        (filters on MP_median absolute value)

    Returns:
        DataFrame with ECB monetary policy shocks

    Example:
        >>> # Get significant policy shocks in 2023
        >>> events = get_monetary_policy_events(
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31",
        ...     shock_threshold=0.05
        ... )
    """
    # Load appropriate frequency
    if frequency.lower() == "daily":
        df = load_ecb_shocks_daily()
    elif frequency.lower() == "monthly":
        df = load_ecb_shocks_monthly()
    else:
        raise ValueError(f"Invalid frequency: {frequency}. Use 'daily' or 'monthly'")

    # Filter by date range
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    # Filter by shock magnitude
    if shock_threshold is not None:
        if 'MP_median' in df.columns:
            df = df[abs(df['MP_median']) >= shock_threshold]
        else:
            logger.warning("[ecb_shocks] MP_median column not found, cannot filter by threshold")

    logger.info(f"[ecb_shocks] Retrieved {len(df)} monetary policy events")
    return df


def classify_shock_type(row: pd.Series) -> str:
    """
    Classify the dominant shock type for a given observation.

    Args:
        row: DataFrame row with MP_median and CBI_median columns

    Returns:
        Shock classification: "dovish_MP", "hawkish_MP", "positive_CBI", "negative_CBI", "mixed"
    """
    mp = row.get('MP_median', 0)
    cbi = row.get('CBI_median', 0)

    # Thresholds
    threshold = 0.02

    if abs(mp) > abs(cbi):
        # Monetary Policy dominant
        return "hawkish_MP" if mp > threshold else "dovish_MP" if mp < -threshold else "neutral_MP"
    else:
        # Central Bank Information dominant
        return "positive_CBI" if cbi > threshold else "negative_CBI" if cbi < -threshold else "neutral_CBI"


def format_for_cognee(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format ECB shocks data for Cognee ingestion with descriptive text.

    Args:
        df: Raw ECB shocks DataFrame

    Returns:
        DataFrame with 'full_text' column for semantic search
    """
    df = df.copy()

    # Add shock classification
    df['shock_type'] = df.apply(classify_shock_type, axis=1)

    df['full_text'] = df.apply(
        lambda row: (
            f"ECB Monetary Policy Event on {row['date'].strftime('%Y-%m-%d')}: "
            f"Policy indicator surprise (pc1): {row.get('pc1', 0):.4f}, "
            f"Monetary Policy shock: {row.get('MP_median', 0):.4f}, "
            f"Central Bank Information shock: {row.get('CBI_median', 0):.4f}. "
            f"Classification: {row.get('shock_type', 'unknown')}. "
            f"Euro Stoxx 50 change: {row.get('STOXX50', 0):.4f}."
        ),
        axis=1
    )

    return df


def get_shocks_for_forex_pair(
        currency_pair: str,
        start_date: str,
        end_date: str,
        frequency: str = "daily"
) -> pd.DataFrame:
    """
    Get ECB monetary policy shocks relevant to a forex pair involving EUR.

    Args:
        currency_pair: Forex pair (must contain EUR, e.g., "EURUSD", "EURGBP")
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        frequency: "daily" or "monthly"

    Returns:
        DataFrame with ECB shocks and currency pair context

    Example:
        >>> df = get_shocks_for_forex_pair("EURUSD", "2023-01-01", "2023-12-31")
    """
    pair_upper = currency_pair.upper()

    if "EUR" not in pair_upper:
        logger.warning(f"[ecb_shocks] Currency pair {currency_pair} does not contain EUR")
        return pd.DataFrame()

    # Load shocks
    df = get_monetary_policy_events(start_date, end_date, frequency)

    if df.empty:
        return df

    # Add currency pair context
    df['currency_pair'] = pair_upper
    df['base_currency'] = pair_upper[:3]
    df['quote_currency'] = pair_upper[3:]

    logger.info(f"[ecb_shocks] Retrieved {len(df)} ECB shocks for {currency_pair}")
    return df
