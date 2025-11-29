from datetime import datetime, timedelta, timezone

from gdelt.alignment import align_candle_to_regime, iter_gdelt_buckets
from gdelt.config import GDELT_TIME_DELTA_MINUTES


def test_align_default_uses_config_bucket():
    candle_time = datetime(2024, 1, 1, 12, 7, tzinfo=timezone.utc)

    aligned = align_candle_to_regime(candle_time)

    assert aligned == datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)


def test_iter_gdelt_buckets_respects_config_bucket_size():
    start = datetime(2024, 1, 1, 12, 7, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 12, 40, tzinfo=timezone.utc)

    buckets = iter_gdelt_buckets(start, end)

    expected_start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    expected = []
    current = expected_start
    while current <= end:
        expected.append(current)
        current += timedelta(minutes=GDELT_TIME_DELTA_MINUTES)

    assert buckets == expected
