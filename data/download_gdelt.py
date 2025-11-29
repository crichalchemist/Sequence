"""
Downloader for GDELT 2.1 GKG files into a local folder.

Example:
  python data/download_gdelt.py --start-date 2016-01-01 --end-date 2016-01-03 --resolution daily --out-dir data/gdelt
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple


BASE_URL = "https://data.gdeltproject.org/gdeltv2"


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)


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
    now = datetime.now(UTC)
    if resolution == "daily":
        return datetime(now.year, now.month, now.day, tzinfo=UTC) - timedelta(seconds=1)
    latest_slot = now - timedelta(minutes=step_minutes)
    return floor_to_step(latest_slot, step_minutes)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _load_http_dependencies() -> Tuple[Any, Any, Any]:
    """
    Import requests and its retry helpers only when available, with an explicit
    error guiding users to install the project's dependencies.
    """

    if importlib.util.find_spec("requests") is None:
        raise ModuleNotFoundError(
            "Missing dependency: requests. Install it with `pip install -r requirements.txt` "
            "before running GDELT downloads."
        )

    requests = importlib.import_module("requests")
    adapters_spec = importlib.util.find_spec("requests.adapters")
    urllib3_retry_spec = importlib.util.find_spec("urllib3.util.retry")
    if adapters_spec is None or urllib3_retry_spec is None:
        missing = []
        if adapters_spec is None:
            missing.append("requests.adapters")
        if urllib3_retry_spec is None:
            missing.append("urllib3.util.retry")
        missing_list = ", ".join(missing)
        raise ModuleNotFoundError(
            f"Missing dependencies: {missing_list}. Install via `pip install -r requirements.txt`."
        )

    adapters = importlib.import_module("requests.adapters")
    urllib3_retry = importlib.import_module("urllib3.util.retry")
    return requests, adapters.HTTPAdapter, urllib3_retry.Retry


def download_file(
    session: Any,
    url: str,
    dest: Path,
    overwrite: bool,
    timeout: int,
    max_retries: int,
    backoff: float,
    expected_checksum: Optional[str],
) -> None:
    if dest.exists() and not overwrite:
        if expected_checksum:
            local_checksum = sha256_file(dest)
            if local_checksum == expected_checksum:
                print(f"[skip] exists {dest.name} (checksum verified)")
                return
            print(
                f"[warn] exists {dest.name} (checksum mismatch: local {local_checksum} vs expected {expected_checksum}); re-downloading"
            )
            dest.unlink()
        else:
            print(f"[skip] exists {dest.name} (no checksum provided)")
            return
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200:
                content = resp.content
                if expected_checksum:
                    checksum = sha256_bytes(content)
                    if checksum != expected_checksum:
                        print(
                            f"[fail] {dest.name} checksum mismatch (got {checksum}, expected {expected_checksum})"
                        )
                        # Treat as retryable corruption rather than accepting bad data.
                        raise ValueError("checksum mismatch")
                    print(f"[ok]   {dest.name} (checksum verified)")
                else:
                    print(f"[ok]   {dest.name} (no checksum provided)")
                dest.write_bytes(content)
                return
            print(f"[warn] {dest.name} -> HTTP {resp.status_code}")
        except Exception as exc:
            print(f"[err]  {dest.name} attempt {attempt}/{max_retries} -> {exc}")
        if attempt < max_retries:
            sleep_time = backoff * attempt
            print(f"[info] retrying {dest.name} in {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    print(f"[fail] {dest.name} after {max_retries} attempts")


def load_checksums(checksum_file: Optional[str]) -> Dict[str, str]:
    if not checksum_file:
        return {}
    path = Path(checksum_file)
    if not path.exists():
        raise FileNotFoundError(f"Checksum file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Checksum file must contain a JSON object mapping filenames to SHA256 values")
    return {str(key): str(value) for key, value in data.items()}


def main():
    requests, HTTPAdapter, Retry = _load_http_dependencies()

    parser = argparse.ArgumentParser(description="Download GDELT 2.1 GKG zip files.")
    parser.add_argument("--start-date", default="2016-01-01", type=parse_date, help="YYYY-MM-DD (UTC) inclusive")
    parser.add_argument("--end-date", default=datetime.now(UTC).strftime("%Y-%m-%d"), type=parse_date, help="YYYY-MM-DD (UTC) inclusive")
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
    parser.add_argument(
        "--checksum-file",
        type=str,
        default=None,
        help="Optional JSON mapping filenames to SHA256 digests for verification",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checksum_map = load_checksums(args.checksum_file)

    start_dt = args.start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = args.end_date.replace(hour=23, minute=59, second=59, microsecond=0)
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
    session.verify = True  # enforce TLS verification
    retry_cfg = Retry(total=0)
    adapter = HTTPAdapter(max_retries=retry_cfg)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    for ts in generator:
        stamp = formatter(ts)
        filename = f"{stamp}.gkg.csv.zip"
        url = f"{BASE_URL}/{filename}"
        dest = out_dir / filename
        expected_checksum = checksum_map.get(filename)
        download_file(
            session,
            url,
            dest,
            overwrite=args.overwrite,
            timeout=args.timeout,
            max_retries=args.max_retries,
            backoff=args.retry_backoff,
            expected_checksum=expected_checksum,
        )


if __name__ == "__main__":
    main()
