"""GDELT processing utilities for regime-aware modeling."""

from gdelt.alignment import align_candle_to_regime
from gdelt.cache import RegimeCache
from gdelt.config import (
    GCAM_FEAR_KEYS,
    GDELT_GKG_BASE_URL,
    REGIME_FEATURE_DIM,
    REGIME_FEATURE_NAMES,
    GDELTThemeConfig,
)
from gdelt.consolidated_downloader import GDELTDownloader
from gdelt.feature_builder import GDELTTimeSeriesBuilder, RegimeFeatureBuilder
from gdelt.parser import GDELTRecord

__all__ = [
    "align_candle_to_regime",
    "RegimeCache",
    "GDELT_GKG_BASE_URL",
    "GCAM_FEAR_KEYS",
    "GDELTThemeConfig",
    "REGIME_FEATURE_DIM",
    "REGIME_FEATURE_NAMES",
    "GDELTDownloader",
    "RegimeFeatureBuilder",
    "GDELTTimeSeriesBuilder",
    "GDELTRecord",
]
