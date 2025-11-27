"""
Helpers to ingest GDELT GKG data and extract per-article tone for sentiment features,
with optional filtering for finance/econ-relevant themes.
"""

from pathlib import Path
from typing import Iterable, Optional, Set

import pandas as pd


# Default finance/econ allowlist (case-sensitive as found in V2Themes)
FINANCE_THEMES: Set[str] = {
    "ECON",
    "ECON_MICRO",
    "ECON_MACRO",
    "ECON_INFLATION",
    "ECON_CENTRALBANK",
    "FINANCE",
    "BANK",
    "STOCK_MARKET",
    "BOND",
    "COMMODITY",
    "OIL",
    "TRADE",
    "CURRENCY",
    "FXRATE",
    "FOREX",
    "GDP",
    "CPI",
    "UNEMPLOYMENT",
    "INTEREST_RATE",
    "MERGER_ACQUISITION",
}


def _parse_themes(theme_str: str) -> Set[str]:
    return {t for t in str(theme_str).split(";") if t}


def _passes_theme_filter(theme_str: str, allowlist: Set[str]) -> bool:
    if not allowlist:
        return True
    themes = _parse_themes(theme_str)
    return not themes.isdisjoint(allowlist)


def load_gdelt_gkg(
    file_path: Path,
    target_tz: Optional[str] = "UTC",
    theme_allowlist: Optional[Iterable[str]] = FINANCE_THEMES,
) -> pd.DataFrame:
    """
    Load a GDELT 2.1 GKG file (tab-delimited, no header) and return a DataFrame with
    datetime, sentiment_score, document_identifier, and V2Themes (filtered).

    Args:
        file_path: path to a GKG file (csv/tsv/gz; pandas will auto-decompress).
        target_tz: timezone to localize/convert. None leaves timestamps naive.
        theme_allowlist: iterable of theme codes to keep; None/empty keeps all.
    """
    # GKG 2.1 column positions (0-based): 1=DATE, 4=DocumentIdentifier, 13=V2Themes, 16=V2Tone
    usecols = [1, 4, 13, 16]
    col_names = ["date", "document_identifier", "v2themes", "v2tone"]
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=col_names,
        usecols=usecols,
        quoting=3,
        on_bad_lines="skip",
        engine="python",
    )

    allowset: Set[str] = set(theme_allowlist) if theme_allowlist else set()
    if allowset:
        df = df[df["v2themes"].apply(lambda t: _passes_theme_filter(t, allowset))]

    # DATE is yyyymmddhhmmss in UTC per GDELT spec.
    df["datetime"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S", errors="coerce")
    if target_tz:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert(target_tz).dt.tz_localize(None)

    # V2Tone is a comma-separated vector; first value is overall tone.
    df["sentiment_score"] = (
        df["v2tone"]
        .astype(str)
        .str.split(",", n=1)
        .str[0]
        .astype(float)
        .where(lambda s: s.notna())
    )

    df = df[["datetime", "sentiment_score", "document_identifier", "v2themes"]].dropna(
        subset=["datetime", "sentiment_score"]
    )
    return df.reset_index(drop=True)
