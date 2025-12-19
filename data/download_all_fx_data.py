import argparse
import csv
import os
from datetime import datetime
from histdata.api import download_hist_data


def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def parse_args():
    parser = argparse.ArgumentParser(description="Download historical FX data for all pairs")
    parser.add_argument("--start-year", type=int, default=2018, help="Start year for downloads")
    parser.add_argument("--end-year", type=int, default=datetime.now().year, help="End year for downloads")
    parser.add_argument("--max-downloads", type=int, default=100, help="Maximum number of downloads to attempt")
    parser.add_argument("--output", type=str, default='output', help="Output directory for data")
    return parser.parse_args()


def _find_pairs_file():
    """Find the pairs.csv file, checking multiple locations."""
    default_pairs = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'pairs.csv'))
    alt_pairs = os.path.join(os.path.dirname(__file__), 'pairs.csv')
    pairs_file = os.environ.get("PAIRS_CSV", default_pairs if os.path.exists(default_pairs) else alt_pairs)
    if not os.path.exists(pairs_file):
        raise FileNotFoundError(f"Pairs file not found at {pairs_file}")
    return pairs_file


def _download_year(year, pair, output_folder):
    """Download data for a single year, trying full year first then monthly."""
    try:
        result = download_hist_data(year=year, pair=pair, output_directory=output_folder, verbose=False)
        print(f'- {year}: {result}')
        return 1  # 1 download
    except AssertionError:
        # Full year failed, try month by month
        return _download_monthly(year, pair, output_folder)


def _download_monthly(year, pair, output_folder):
    """Download data month by month for a given year."""
    downloads = 0
    for month in range(1, 13):
        try:
            result = download_hist_data(
                year=str(year), month=str(month), pair=pair,
                output_directory=output_folder, verbose=False
            )
            print(f'- {year}-{month:02d}: {result}')
            downloads += 1
        except Exception as e:
            print(f"Failed to download {pair} {year}-{month}: {e}")
            break  # Stop trying months for this year if one fails
    return downloads


def _download_pair(currency_pair_name, pair, history_first_trading_month, args, max_downloads, current_downloads):
    """Download all data for a single currency pair."""
    start_year = int(history_first_trading_month[0:4])
    end_year = min(args.end_year, datetime.now().year)
    
    print(f"Downloading {currency_pair_name} ({pair}) from {start_year} to {end_year}")
    output_folder = os.path.join(args.output, pair)
    mkdir_p(output_folder)
    
    downloads = 0
    try:
        for year in range(start_year, end_year + 1):
            if current_downloads + downloads >= max_downloads:
                print(f"Reached max downloads limit ({max_downloads}). Stopping.")
                break
            
            year_downloads = _download_year(year, pair, output_folder)
            downloads += year_downloads
            
            if current_downloads + downloads >= max_downloads:
                break
                
    except Exception as e:
        print(f'[DONE] for currency {currency_pair_name} - error: {e}')
    
    return downloads


def download_all(args):
    """Download FX data with proper bounds to prevent infinite loops."""
    pairs_file = _find_pairs_file()
    total_downloads = 0
    max_downloads = args.max_downloads
    
    with open(pairs_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            if total_downloads >= max_downloads:
                print(f"Reached max downloads limit ({max_downloads}). Stopping.")
                break
            
            currency_pair_name, pair, history_first_trading_month = row
            pair_downloads = _download_pair(
                currency_pair_name, pair, history_first_trading_month,
                args, max_downloads, total_downloads
            )
            total_downloads += pair_downloads
            
            print(f"Completed {currency_pair_name}. Total downloads so far: {total_downloads}")
    
    print(f"Download complete. Total downloads: {total_downloads}")


if __name__ == '__main__':
    args = parse_args()
    download_all(args)
