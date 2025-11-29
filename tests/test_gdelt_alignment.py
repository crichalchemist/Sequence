import importlib.util
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


ROOT = Path(__file__).resolve().parents[1]

gdelt_pkg = types.ModuleType("gdelt")
gdelt_pkg.__path__ = [str(ROOT / "gdelt")]
sys.modules.setdefault("gdelt", gdelt_pkg)

config = _load_module("gdelt.config", ROOT / "gdelt" / "config.py")
setattr(sys.modules["gdelt"], "config", config)
alignment = _load_module("gdelt.alignment", ROOT / "gdelt" / "alignment.py")

align_candle_to_regime = alignment.align_candle_to_regime
iter_gdelt_buckets = alignment.iter_gdelt_buckets
GDELT_TIME_DELTA_MINUTES = config.GDELT_TIME_DELTA_MINUTES
get_gdelt_bucket_minutes = config.get_gdelt_bucket_minutes


def test_align_default_uses_config_bucket():
    bucket = get_gdelt_bucket_minutes()
    candle_time = datetime(2024, 1, 1, 10, 16, 45, tzinfo=timezone.utc)

    aligned = align_candle_to_regime(candle_time)

    expected_minute = (candle_time.minute // bucket) * bucket
    expected = candle_time.replace(minute=expected_minute, second=0, microsecond=0)
    assert aligned == expected
    assert bucket == GDELT_TIME_DELTA_MINUTES


def test_iter_gdelt_buckets_matches_alignment():
    bucket = get_gdelt_bucket_minutes()
    start = datetime(2024, 1, 1, 10, 2, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 10, 50, tzinfo=timezone.utc)

    buckets = iter_gdelt_buckets(start, end)

    assert buckets[0] == align_candle_to_regime(start, bucket_minutes=bucket)
    assert buckets[-1] == datetime(2024, 1, 1, 10, 45, tzinfo=timezone.utc)
    assert all(
        b2 - b1 == timedelta(minutes=bucket) for b1, b2 in zip(buckets, buckets[1:])
    )
    assert bucket == GDELT_TIME_DELTA_MINUTES
