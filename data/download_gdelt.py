"""
Downloader for GDELT 2.1 GKG files into a local folder.

Example:
  python data/download_gdelt.py --start-date 2016-01-01 --end-date 2016-01-03 --resolution daily --out-dir data/gdelt
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import requests


BASE_URL = "http://data.gdeltproject.org/gdeltv2"


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def iter_timestamps(start: datetime, end: datetime, step_minutes: int = 15) -> Iterator[datetime]:
    current = start
    delta = timedelta(minutes=step_minutes)
    while current <= end:
        yield current
        current += delta


def iter_days(start: datetime, end: datetime) -> Iterator[datetime]:
    current = start
    delta = timedelta(days=1)
    while current <= end:
        yield current
        current += delta


def download_file(url: str, dest: Path, overwrite: bool, timeout: int = 30) -> None:
    if dest.exists() and not overwrite:
        print(f"[skip] exists {dest.name}")
        return
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            dest.write_bytes(resp.content)
            print(f"[ok]   {dest.name}")
        else:
            print(f"[warn] {dest.name} -> HTTP {resp.status_code}")
    except Exception as exc:
        print(f"[err]  {dest.name} -> {exc}")


def main():
    parser = argparse.ArgumentParser(description="Download GDELT 2.1 GKG zip files.")
    parser.add_argument("--start-date", default="2016-01-01", type=parse_date, help="YYYY-MM-DD (UTC) inclusive")
    parser.add_argument("--end-date", default=datetime.utcnow().strftime("%Y-%m-%d"), type=parse_date, help="YYYY-MM-DD (UTC) inclusive")
    parser.add_argument(
        "--resolution",
        choices=["daily", "15min"],
        default="daily",
        help="daily = one file per day; 15min = full 15-minute stream",
    )
    parser.add_argument("--step-minutes", type=int, default=15, help="Cadence between files when --resolution=15min")
    parser.add_argument("--out-dir", default="data/gdelt", help="Destination directory for zip files")
    parser.add_argument("--overwrite", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime(args.start_date.year, args.start_date.month, args.start_date.day)
    end_dt = datetime(args.end_date.year, args.end_date.month, args.end_date.day, 23, 59, 59)

    if args.resolution == "daily":
        generator = iter_days(start_dt, end_dt)
        formatter = lambda dt: dt.strftime("%Y%m%d")
    else:
        generator = iter_timestamps(start_dt, end_dt, step_minutes=args.step_minutes)
        formatter = lambda dt: dt.strftime("%Y%m%d%H%M%S")

    for ts in generator:
        stamp = formatter(ts)
        filename = f"{stamp}.gkg.csv.zip"
        url = f"{BASE_URL}/{filename}"
        dest = out_dir / filename
        download_file(url, dest, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
