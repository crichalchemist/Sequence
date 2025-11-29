"""Tests for GDELT alignment utilities."""
from datetime import datetime, timezone
from importlib import util
from pathlib import Path
import sys
import types

def _load_config_module():
    config_path = Path(__file__).resolve().parents[1] / "gdelt" / "config.py"
    spec = util.spec_from_file_location("gdelt.config", config_path)
    assert spec and spec.loader
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[spec.name] = module
    return module


def _load_alignment_module():
    if "gdelt" not in sys.modules:
        gdelt_module = types.ModuleType("gdelt")
        gdelt_module.__path__ = [str(Path(__file__).resolve().parents[1] / "gdelt")]
        sys.modules["gdelt"] = gdelt_module
    _load_config_module()
    alignment_path = Path(__file__).resolve().parents[1] / "gdelt" / "alignment.py"
    spec = util.spec_from_file_location("gdelt.alignment", alignment_path)
    assert spec and spec.loader
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_align_defaults_to_config_bucket_minutes() -> None:
    alignment = _load_alignment_module()
    config = _load_config_module()
    ts = datetime(2024, 1, 1, 0, 23, tzinfo=timezone.utc)

    aligned = alignment.align_candle_to_regime(ts)

    assert aligned == datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc)
    assert alignment.get_gdelt_bucket_minutes() == config.GDELT_TIME_DELTA_MINUTES


def test_iter_gdelt_buckets_respects_configured_bucket_size() -> None:
    alignment = _load_alignment_module()
    start = datetime(2024, 1, 1, 0, 7, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 0, 52, tzinfo=timezone.utc)

    buckets = alignment.iter_gdelt_buckets(start, end)

    assert buckets == [
        datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 0, 45, tzinfo=timezone.utc),
    ]
