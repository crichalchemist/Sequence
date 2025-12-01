"""
Controlled HistData backfill with bounded years and download quotas.

The previous implementation looped indefinitely, relying on exceptions to stop
downloads. This version adds explicit year bounds, per-run quotas, and
structured logging to make the ingestion safer to operate.
"""

import argparse
import csv
import logging
import os
from typing import List, Tuple

from histdata.api import download_hist_data


LOGGER = logging.getLogger(__name__)


def mkdir_p(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_pairs_file(pairs_file: str) -> List[Tuple[str, str, int]]:
    """Return (friendly_name, pair, start_year) tuples from pairs.csv."""

    with open(pairs_file, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=",")
        next(reader, None)  # skip header
        parsed_rows: List[Tuple[str, str, int]] = []
        for row in reader:
            if len(row) < 3:
                LOGGER.warning("Skipping malformed pairs row: %s", row)
                continue
            currency_pair_name, pair, history_first_trading_month = row
            parsed_rows.append((currency_pair_name, pair, int(history_first_trading_month[0:4])))
    return parsed_rows


def clamp_year_range(start_year: int, end_year: int, earliest_year: int) -> Tuple[int, int]:
    bounded_start = max(start_year, earliest_year)
    bounded_end = max(bounded_start, end_year)
    if bounded_start != start_year:
        LOGGER.info(
            "Raising start_year to %s to respect earliest available data (requested %s)",
            bounded_start,
            start_year,
        )
    if bounded_end != end_year:
        LOGGER.info("Raising end_year to %s because start_year is later than requested end_year %s", bounded_end, end_year)
    return bounded_start, bounded_end


def download_year(pair: str, year: int, output_directory: str) -> int:
    """Attempt a full-year download; fall back to months. Returns files downloaded."""

    LOGGER.info("Downloading %s for %s", year, pair)
    try:
        download_hist_data(year=year, pair=pair, output_directory=output_directory, verbose=False)
        return 1
    except AssertionError:
        LOGGER.info("Full-year archive unavailable for %s %s; falling back to monthly downloads", pair, year)

    downloaded = 0
    for month in range(1, 13):
        try:
            download_hist_data(year=str(year), month=str(month), pair=pair, output_directory=output_directory, verbose=False)
            downloaded += 1
        except Exception as exc:  # noqa: BLE001 - logging unexpected provider errors is useful context
            LOGGER.warning("Skipping %s %s-%02d due to error: %s", pair, year, month, exc)
    return downloaded


def download_pair_range(
    pair: str,
    start_year: int,
    end_year: int,
    output_directory: str,
    max_downloads: int,
) -> int:
    downloaded = 0
    for year in range(start_year, end_year + 1):
        if downloaded >= max_downloads:
            LOGGER.warning("Reached max_downloads=%s; stopping further downloads for %s", max_downloads, pair)
            break
        downloaded += download_year(pair, year, output_directory)
    return downloaded


def download_all(
    *,
    output: str,
    pairs_file: str,
    start_year: int,
    end_year: int,
    max_downloads: int,
):
    pairs = parse_pairs_file(pairs_file)
    if not pairs:
        raise ValueError(f"No pairs found in {pairs_file}")

    earliest_year = min(row[2] for row in pairs)
    start_year, end_year = clamp_year_range(start_year, end_year, earliest_year)

    for currency_pair_name, pair, first_year in pairs:
        pair_start, pair_end = clamp_year_range(start_year, end_year, first_year)
        output_folder = os.path.join(output, pair)
        mkdir_p(output_folder)
        LOGGER.info("[%s] Downloading %s through %s into %s", pair, pair_start, pair_end, output_folder)
        downloaded = download_pair_range(pair, pair_start, pair_end, output_folder, max_downloads)
        LOGGER.info("[%s] Downloaded %s archives", pair, downloaded)


def resolve_pairs_file() -> str:
    # Prefer repo-root pairs.csv to keep full symbol coverage; allow override via env.
    default_pairs = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "pairs.csv"))
    alt_pairs = os.path.join(os.path.dirname(__file__), "pairs.csv")
    return os.environ.get("PAIRS_CSV", default_pairs if os.path.exists(default_pairs) else alt_pairs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded HistData downloader with quotas and logging")
    parser.add_argument("--output", default=os.environ.get("FX_DATA_OUTPUT", "output"), help="Target directory for downloads")
    parser.add_argument("--pairs-file", default=resolve_pairs_file(), help="Path to pairs.csv (defaults to repo root)")
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="First year to download (defaults to latest start year in pairs.csv)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last year to download inclusive (defaults to start-year)",
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=120,
        help="Per-pair cap on archives downloaded in a single run (yearly counts as 1)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    pairs_file = args.pairs_file
    if not os.path.exists(pairs_file):
        raise FileNotFoundError(f"Pairs file not found at {pairs_file}")

    pairs = parse_pairs_file(pairs_file)
    if not pairs:
        raise ValueError(f"No pairs found in {pairs_file}")

    if args.max_downloads <= 0:
        raise ValueError("--max-downloads must be a positive integer")

    default_start_year = max(row[2] for row in pairs)
    start_year = args.start_year or default_start_year
    end_year = args.end_year or start_year

    download_all(
        output=args.output,
        pairs_file=pairs_file,
        start_year=start_year,
        end_year=end_year,
        max_downloads=args.max_downloads,
    )


if __name__ == "__main__":
    main()
