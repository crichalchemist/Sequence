"""Downloader for GDELT GKG files."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import requests

from gdelt.config import GDELT_GKG_BASE_URL, GDELT_GKG_EXTENSION, GDELT_TIME_DELTA_MINUTES

logger = logging.getLogger(__name__)


class GDELTDownloader:
    """Download raw GDELT GKG files for a datetime range."""

    def __init__(
        self,
        output_dir: Path | str = Path("data/gdelt_raw"),
        session: Optional[requests.Session] = None,
        timeout: int = 15,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.timeout = timeout

    def fetch_gkg_files(self, start_dt: datetime, end_dt: datetime) -> List[Path]:
        """
        Download GDELT GKG files between ``start_dt`` and ``end_dt`` (inclusive).

        The GDELT feed updates every 15 minutes; timestamps are floored to the
        nearest available bucket to avoid 404s.
        """

        if end_dt < start_dt:
            raise ValueError("end_dt must be after start_dt")

        start = self._floor_to_bucket(start_dt)
        end = self._floor_to_bucket(end_dt)
        current = start
        downloaded: List[Path] = []

        while current <= end:
            path = self._download_single(current)
            if path:
                downloaded.append(path)
            current += timedelta(minutes=GDELT_TIME_DELTA_MINUTES)

        return downloaded

    def _download_single(self, ts: datetime) -> Optional[Path]:
        ts = ts.astimezone(timezone.utc)
        timestamp = ts.strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}{GDELT_GKG_EXTENSION}"
        url = f"{GDELT_GKG_BASE_URL}/{filename}"
        target_path = self.output_dir / filename

        if target_path.exists():
            logger.debug("GDELT file already present: %s", target_path)
            return target_path

        logger.info("Downloading GDELT GKG: %s", url)
        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code == 404:
            logger.warning("GDELT file missing for %s", timestamp)
            return None
        resp.raise_for_status()

        target_path.write_bytes(resp.content)
        return target_path

    @staticmethod
    def _floor_to_bucket(dt: datetime) -> datetime:
        dt = dt.astimezone(timezone.utc)
        minute_bucket = (dt.minute // GDELT_TIME_DELTA_MINUTES) * GDELT_TIME_DELTA_MINUTES
        return dt.replace(minute=minute_bucket, second=0, microsecond=0)
