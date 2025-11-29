"""GDELT processing utilities for regime-aware modeling."""

from gdelt.alignment import align_candle_to_regime
from gdelt.cache import RegimeCache
from gdelt.config import (
    GDELT_GKG_BASE_URL,
    GCAM_FEAR_KEYS,
    GDELTThemeConfig,
    REGIME_FEATURE_DIM,
    REGIME_FEATURE_NAMES,
)
from gdelt.downloader import GDELTDownloader
from gdelt.feature_builder import RegimeFeatureBuilder
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
    "GDELTRecord",
]
