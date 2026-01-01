"""Download historical crypto OHLCV data from Coinbase.

This script downloads minute-level OHLCV data for cryptocurrency pairs from Coinbase
and saves in the same format as FX data for consistency.

Usage:
    python data/download_crypto_data.py --start-year 2018 --end-year 2024
    python data/download_crypto_data.py --pairs BTC-USD,ETH-USD --output data/crypto
"""

import argparse
import csv
# Add project root to path
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)

# Coinbase Advanced Trade API endpoints
COINBASE_BASE_URL = "https://api.exchange.coinbase.com"
COINBASE_CANDLES_ENDPOINT = "/products/{product_id}/candles"

# Rate limiting
MAX_REQUESTS_PER_SECOND = 10  # Coinbase public endpoint limit
REQUEST_DELAY = 1.0 / MAX_REQUESTS_PER_SECOND  # 0.1 seconds

# Interval to granularity mapping (in seconds)
INTERVAL_TO_GRANULARITY = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "1d": 86400
}


class CoinbaseCryptoDownloader:
    """Download historical OHLCV data from Coinbase."""

    def __init__(self, output_root: Path = Path("data/crypto")):
        """
        Args:
            output_root: Root directory for downloaded data
        """
        self.output_root = output_root
        self.session = requests.Session()
        self.requests_made = 0

    def _rate_limit(self):
        """Apply rate limiting to avoid hitting Coinbase limits."""
        time.sleep(REQUEST_DELAY)
        self.requests_made += 1

        if self.requests_made % 50 == 0:
            logger.info(f"Made {self.requests_made} API requests")

    def download_candles(
            self,
            product_id: str,
            granularity: int,
            start_time: int,
            end_time: int
    ) -> list[list]:
        """Download candles (candlestick data) from Coinbase.

        Args:
            product_id: Coinbase product ID (e.g., 'BTC-USD')
            granularity: Candle size in seconds (60, 300, 900, 3600, 86400)
            start_time: Start timestamp in seconds (UNIX timestamp)
            end_time: End timestamp in seconds (UNIX timestamp)

        Returns:
            List of candles data: [timestamp, low, high, open, close, volume]
        """
        url = f"{COINBASE_BASE_URL}{COINBASE_CANDLES_ENDPOINT.format(product_id=product_id)}"

        params = {
            "granularity": granularity,
            "start": start_time,
            "end": end_time
        }

        self._rate_limit()

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download candles for {product_id}: {e}")
            return []

    def download_year(
            self,
            pair: str,
            year: int,
            interval: str = "1m"
    ) -> pd.DataFrame | None:
        """Download full year of data for a trading pair.

        Args:
            pair: Trading pair (e.g., 'BTC-USD')
            year: Year to download
            interval: Timeframe (1m, 5m, 1h, 1d)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        # Coinbase uses product IDs directly (e.g., 'BTC-USD')
        product_id = pair.upper()
        granularity = INTERVAL_TO_GRANULARITY[interval]

        # Calculate year boundaries in seconds (UNIX timestamp)
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)

        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())

        logger.info(f"Downloading {pair} for year {year}")

        all_candles = []
        current_start = start_ts

        # Coinbase returns max 300 candles per request
        # For 1-minute data: 300 minutes = 5 hours
        # We need to make multiple requests to cover a full year
        max_candles_per_request = 300
        batch_duration = granularity * max_candles_per_request

        while current_start < end_ts:
            # Calculate end time for this batch
            batch_end = min(current_start + batch_duration, end_ts)

            candles = self.download_candles(
                product_id=product_id,
                granularity=granularity,
                start_time=current_start,
                end_time=batch_end
            )

            if not candles:
                # Move forward anyway to avoid infinite loop
                current_start = batch_end
                continue

            all_candles.extend(candles)

            # Move to next batch
            current_start = batch_end

            # Log progress
            current_date = datetime.fromtimestamp(current_start)
            logger.info(f"  Downloaded up to {current_date.strftime('%Y-%m-%d %H:%M')}")

        if not all_candles:
            logger.warning(f"No data downloaded for {pair} in {year}")
            return None

        # Convert to DataFrame
        # Coinbase format: [timestamp, low, high, open, close, volume]
        df = pd.DataFrame(all_candles, columns=[
            'timestamp', 'low', 'high', 'open', 'close', 'volume'
        ])

        # Convert timestamp to datetime (Coinbase returns seconds)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Convert numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by timestamp (ascending order)
        df = df.sort_values('timestamp')

        # Remove duplicates (can happen at batch boundaries)
        df = df.drop_duplicates(subset=['timestamp'])

        logger.info(f"Downloaded {len(df):,} candles for {pair} in {year}")

        return df

    def save_to_csv(self, df: pd.DataFrame, output_path: Path):
        """Save DataFrame to CSV in HistData-compatible format.

        Args:
            df: DataFrame with OHLCV data
            output_path: Path to save CSV file
        """
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to HistData format: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
        df_export = df.copy()
        df_export['DateTime'] = df_export['timestamp'].dt.strftime('%Y%m%d %H%M%S')

        # Select and rename columns
        df_export = df_export[[
            'DateTime', 'open', 'high', 'low', 'close', 'volume'
        ]]

        # Rename to match HistData format
        df_export.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']

        # Save to CSV with semicolon delimiter (HistData format)
        df_export.to_csv(
            output_path,
            sep=';',
            index=False,
            header=False  # HistData has no header
        )

        logger.info(f"Saved {len(df_export):,} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download historical crypto data from Coinbase")
    parser.add_argument(
        "--start-year",
        type=int,
        default=2017,
        help="Start year for downloads"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=datetime.now().year,
        help="End year for downloads"
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=Path("crypto_pairs.csv"),
        help="CSV file with crypto pairs to download"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/crypto"),
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1m",
        choices=["1m", "5m", "15m", "1h", "1d"],
        help="Timeframe for candles"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("CRYPTO DATA DOWNLOAD FROM COINBASE")
    logger.info("=" * 80)
    logger.info(f"Years: {args.start_year} - {args.end_year}")
    logger.info(f"Pairs CSV: {args.pairs_csv}")
    logger.info(f"Interval: {args.interval}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 80)

    # Check if pairs CSV exists
    if not args.pairs_csv.exists():
        logger.error(f"Pairs CSV not found: {args.pairs_csv}")
        return 1

    # Read crypto pairs
    crypto_pairs = []
    with open(args.pairs_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            crypto_pairs.append({
                'pair': row['pair'],
                'name': row['currency_pair_name'],
                'first_trading_month': row['history_first_trading_month']
            })

    logger.info(f"Found {len(crypto_pairs)} crypto pairs to download")

    # Initialize downloader
    downloader = CoinbaseCryptoDownloader(output_root=args.output)

    # Download each pair
    for pair_info in crypto_pairs:
        pair = pair_info['pair']
        pair_name = pair_info['name']

        # Determine start year (use max of args.start_year and first trading month)
        first_trading_year = int(pair_info['first_trading_month'].split('-')[0])
        start_year = max(args.start_year, first_trading_year)

        logger.info(f"\nDownloading {pair_name} ({pair}) from {start_year} to {args.end_year}")

        pair_lower = pair.replace('-', '').lower()
        output_dir = args.output / pair_lower

        for year in range(start_year, args.end_year + 1):
            # Download year of data
            df = downloader.download_year(pair, year, interval=args.interval)

            if df is not None:
                # Save to CSV
                output_file = output_dir / f"{pair_lower}_{year}.csv"
                downloader.save_to_csv(df, output_file)
            else:
                logger.warning(f"Skipping {pair} for year {year} - no data available")

    logger.info("\n" + "=" * 80)
    logger.info("✅ CRYPTO DATA DOWNLOAD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Data saved to: {args.output}")
    logger.info(f"Total API requests: {downloader.requests_made}")

    return 0


if __name__ == "__main__":
    exit(main())
