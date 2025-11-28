"""
Downloader for GDELT 2.1 GKG files into a local folder.

Example:
  python data/download_gdelt.py --start-date 2016-01-01 --end-date 2016-01-03 --resolution daily --out-dir data/gdelt
"""

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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


def floor_to_step(dt: datetime, step_minutes: int) -> datetime:
    return dt - timedelta(
        minutes=dt.minute % step_minutes,
        seconds=dt.second,
        microseconds=dt.microsecond,
    )


def latest_available_end(resolution: str, step_minutes: int) -> datetime:
    """
    GDELT only publishes 15-minute batches a few minutes after the window ends
    and daily files the following day. Cap requests to avoid 404s on future windows.
    """
    now = datetime.utcnow()
    if resolution == "daily":
        return datetime(now.year, now.month, now.day) - timedelta(seconds=1)
    latest_slot = now - timedelta(minutes=step_minutes)
    return floor_to_step(latest_slot, step_minutes)


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    overwrite: bool,
    timeout: int,
    max_retries: int,
    backoff: float,
) -> None:
    if dest.exists() and not overwrite:
        print(f"[skip] exists {dest.name}")
        return
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200:
                dest.write_bytes(resp.content)
                print(f"[ok]   {dest.name}")
                return
            print(f"[warn] {dest.name} -> HTTP {resp.status_code}")
        except Exception as exc:
            print(f"[err]  {dest.name} attempt {attempt}/{max_retries} -> {exc}")
        if attempt < max_retries:
            sleep_time = backoff * attempt
            print(f"[info] retrying {dest.name} in {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    print(f"[fail] {dest.name} after {max_retries} attempts")


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
    parser.add_argument("--timeout", type=int, default=10, help="HTTP timeout per request (seconds)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max attempts per file before giving up")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Seconds multiplied by attempt number between retries")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime(args.start_date.year, args.start_date.month, args.start_date.day)
    end_dt = datetime(args.end_date.year, args.end_date.month, args.end_date.day, 23, 59, 59)
    max_end_dt = latest_available_end(args.resolution, args.step_minutes)

    if args.resolution == "15min":
        start_dt = floor_to_step(start_dt, args.step_minutes)
        end_dt = floor_to_step(end_dt, args.step_minutes)

    if end_dt > max_end_dt:
        print(f"[info] Clamping end datetime to last available window {max_end_dt} UTC to avoid 404s")
        end_dt = max_end_dt

    if start_dt > end_dt:
        print(f"[warn] Start datetime {start_dt} is after available data ({end_dt}); nothing to download.")
        return

    if args.resolution == "daily":
        generator = iter_days(start_dt, end_dt)
        formatter = lambda dt: dt.strftime("%Y%m%d")
    else:
        generator = iter_timestamps(start_dt, end_dt, step_minutes=args.step_minutes)
        formatter = lambda dt: dt.strftime("%Y%m%d%H%M%S")

    session = requests.Session()
    retry_cfg = Retry(total=0)
    adapter = HTTPAdapter(max_retries=retry_cfg)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    for ts in generator:
        stamp = formatter(ts)
        filename = f"{stamp}.gkg.csv.zip"
        url = f"{BASE_URL}/{filename}"
        dest = out_dir / filename
        download_file(
            session,
            url,
            dest,
            overwrite=args.overwrite,
            timeout=args.timeout,
            max_retries=args.max_retries,
            backoff=args.retry_backoff,
        )


if __name__ == "__main__":
    main()
