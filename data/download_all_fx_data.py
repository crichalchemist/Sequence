import csv
import os

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


def download_all():
    output = os.environ.get("FX_DATA_OUTPUT", 'output')

    # Prefer repo-root pairs.csv to keep full symbol coverage; allow override via env.
    default_pairs = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'pairs.csv'))
    alt_pairs = os.path.join(os.path.dirname(__file__), 'pairs.csv')
    pairs_file = os.environ.get("PAIRS_CSV", default_pairs if os.path.exists(default_pairs) else alt_pairs)
    if not os.path.exists(pairs_file):
        raise FileNotFoundError(f"Pairs file not found at {pairs_file}")
    with open(pairs_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            currency_pair_name, pair, history_first_trading_month = row
            year = int(history_first_trading_month[0:4])
            print(currency_pair_name)
            output_folder = os.path.join(output, pair)
            mkdir_p(output_folder)
            try:
                while True:
                    could_download_full_year = False
                    try:
                        print('-', download_hist_data(year=year,
                                                      pair=pair,
                                                      output_directory=output_folder,
                                                      verbose=False))
                        could_download_full_year = True
                    except AssertionError:
                        pass  # lets download it month by month.
                    month = 1
                    while not could_download_full_year and month <= 12:
                        print('-', download_hist_data(year=str(year),
                                                      month=str(month),
                                                      pair=pair,
                                                      output_directory=output_folder,
                                                      verbose=False))
                        month += 1
                    year += 1
            except Exception:
                print('[DONE] for currency', currency_pair_name)


if __name__ == '__main__':
    download_all()
