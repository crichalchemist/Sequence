"""Utilities for aligning candle times to GDELT regime buckets."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gdelt.config import GDELT_TIME_DELTA_MINUTES


def align_candle_to_regime(
    candle_time: datetime, bucket_minutes: int = GDELT_TIME_DELTA_MINUTES
) -> datetime:
    """Floor candle_time to the previous full regime bucket using the configured default interval.

    The default bucket size is defined by :data:`GDELT_TIME_DELTA_MINUTES` to keep
    regime alignment consistent across ingestion utilities.
    """
    candle_time = candle_time.astimezone(timezone.utc)
    minutes = (candle_time.minute // bucket_minutes) * bucket_minutes
    aligned = candle_time.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minutes)
    if aligned > candle_time:
        aligned -= timedelta(minutes=bucket_minutes)
    return aligned


def iter_gdelt_buckets(start: datetime, end: datetime) -> list[datetime]:
    """Generate regime bucket start times between start and end inclusive.

    Uses the configured bucket size to keep iteration and alignment in sync.
    """
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    if end < start:
        raise ValueError("end must be after start")
    buckets = []
    current = align_candle_to_regime(start, bucket_minutes=GDELT_TIME_DELTA_MINUTES)
    while current <= end:
        buckets.append(current)
        current += timedelta(minutes=GDELT_TIME_DELTA_MINUTES)
    return buckets
