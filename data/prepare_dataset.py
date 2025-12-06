"""
Utility to load HistData CSVs (Central time), build features, and create train/val/test datasets.

Example:
  PYTHONPATH=.venv/bin/python data/prepare_dataset.py --pair gbpusd --t-in 120 --t-out 10 --target-type classification
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local utilities – imported after ensuring the repository root is on ``sys.path``.
from utils.logger import get_logger
from utils.datetime_utils import convert_to_utc_and_dedup
from utils.seed import set_seed

logger = get_logger(__name__)

from config.config import DataConfig, FeatureConfig
from features.agent_features import build_feature_frame
from features.agent_sentiment import aggregate_sentiment, attach_sentiment_features
from features.intrinsic_time import build_intrinsic_time_bars


def validate_dataframe(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """Validate and sanitize raw price data."""
    logger.info(f"Validating DataFrame with {len(df)} rows and columns: {list(df.columns)}")
    
    # Check required columns
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Check and convert datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        logger.warning("Converting datetime column to datetime64[ns]")
        df["datetime"] = pd.to_datetime(df["datetime"])
    
    # Remove NaN rows for critical columns
    initial_len = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} NaN rows from OHLC data")
    
    # Detect and remove duplicate timestamps
    dups = df.duplicated(subset=["datetime"], keep="first")
    if dups.any():
        logger.warning(f"Removing {dups.sum()} duplicate timestamps")
        df = df[~dups].reset_index(drop=True)
    
    # Validate OHLC relationships
    invalid_ohlc = (df["high"] < df["low"]) | (df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["low"] > df["open"]) | (df["low"] > df["close"])
    if invalid_ohlc.any():
        invalid_count = invalid_ohlc.sum()
        logger.warning(f"Removing {invalid_count} rows with invalid OHLC relationships")
        df = df[~invalid_ohlc].reset_index(drop=True)
    
    # Sort by datetime and reset index
    df = df.sort_values("datetime").reset_index(drop=True)
    
    logger.info(f"Validation complete. DataFrame reduced to {len(df)} valid rows")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare datasets from Central-time HistData zips.")
    parser.add_argument("--pairs", default="gbpusd", help="Comma-separated pair codes (e.g., gbpusd,eurusd)")
    parser.add_argument("--input-root", default="output_central", help="Root directory containing pair subfolders with zips")
    parser.add_argument("--years", default=None, help="Comma-separated list of years to include (e.g., 2018,2019). Default: all")
    parser.add_argument("--t-in", type=int, default=120, help="Lookback window length")
    parser.add_argument("--t-out", type=int, default=10, help="Forecast horizon in minutes")
    parser.add_argument("--lookahead-window", type=int, default=None, help="Lookahead window for auxiliary targets")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K future returns/prices to supervise")
    parser.add_argument("--predict-sell-now", action="store_true", help="Whether to create a sell-now classification label")
    parser.add_argument("--target-type", choices=["classification", "regression"], default="classification")
    parser.add_argument("--flat-threshold", type=float, default=0.0001, help="Flat class threshold for log returns")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train fraction (time-ordered)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation fraction (time-ordered)")
    parser.add_argument("--feature-groups", default="all", help="Comma-separated feature groups to include or 'all'")
    parser.add_argument("--exclude-feature-groups", default=None, help="Comma-separated feature groups to drop")
    parser.add_argument("--sma-windows", default="10,20,50", help="Comma-separated SMA window lengths")
    parser.add_argument("--ema-windows", default="10,20,50", help="Comma-separated EMA spans")
    parser.add_argument("--rsi-window", type=int, default=14, help="Window length for RSI")
    parser.add_argument("--bollinger-window", type=int, default=20, help="Window length for Bollinger bands")
    parser.add_argument("--bollinger-num-std", type=float, default=2.0, help="Std dev multiplier for Bollinger bands")
    parser.add_argument("--atr-window", type=int, default=14, help="Window length for ATR")
    parser.add_argument("--short-vol-window", type=int, default=10, help="Short window for volatility clustering")
    parser.add_argument("--long-vol-window", type=int, default=50, help="Long window for volatility clustering")
    parser.add_argument("--spread-windows", default="20", help="Comma-separated windows for normalized spread stats")
    parser.add_argument("--imbalance-smoothing", type=int, default=5, help="Rolling mean window for wick/body imbalance")
    parser.add_argument(
        "--intrinsic-time",
        action="store_true",
        help="Convert minute bars to intrinsic-time bars via directional-change events.",
    )
    parser.add_argument(
        "--dc-threshold-up",
        type=float,
        default=0.001,
        help="Fractional increase needed to flag an upward directional change (e.g., 0.001=0.1%).",
    )
    parser.add_argument(
        "--dc-threshold-down",
        type=float,
        default=None,
        help="Fractional decrease needed to flag a downward directional change. Defaults to dc-threshold-up.",
    )
    parser.add_argument(
        "--include-sentiment",
        action="store_true",
        help="Include news sentiment features (requires GDELT data). Adds sentiment_score, sentiment_5min, sentiment_15min, sentiment_60min columns.",
    )
    return parser.parse_args()


def _load_pair_data(pair: str, input_root: Path, years: Optional[List[str]]) -> pd.DataFrame:
    pair_dir = input_root / pair
    if not pair_dir.exists():
        raise FileNotFoundError(f"No data folder for pair {pair} under {input_root}")

    zips = sorted(pair_dir.glob("*.zip"))
    csvs = sorted(pair_dir.glob("*.csv"))
    if years:
        zips = [z for z in zips if any(y in z.name for y in years)]
        csvs = [c for c in csvs if any(y in c.name for y in years)]
    if not zips and not csvs:
        raise FileNotFoundError(f"No data files found for pair {pair} with years={years}")

    frames = []
    for zpath in zips:
        with ZipFile(zpath) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                continue
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(
                    f,
                    sep=";",
                    header=None,
                    names=["datetime", "open", "high", "low", "close", "volume"],
                )
                df["source_file"] = zpath.name
                frames.append(df)
    for cpath in csvs:
        try:
            df = pd.read_csv(cpath)
            if {"datetime", "open", "high", "low", "close", "volume"}.issubset(df.columns):
                df = df[["datetime", "open", "high", "low", "close", "volume"]]
            else:
                # Fallback to legacy format without header; assume comma-delimited.
                df = pd.read_csv(
                    cpath,
                    header=None,
                    names=["datetime", "open", "high", "low", "close", "volume"],
                )
            df["source_file"] = cpath.name
            frames.append(df)
        except Exception as exc:
            logger.warning("[warn] failed to load CSV %s: %s", cpath, exc)
    if not frames:
        raise RuntimeError("No CSV data loaded; check zip contents.")

    full_df = pd.concat(frames, ignore_index=True)
    full_df["datetime"] = pd.to_datetime(full_df["datetime"], format="%Y%m%d %H%M%S")
    full_df = full_df.sort_values("datetime").reset_index(drop=True)
    
    # Validate the data after loading
    required_cols = ["datetime", "open", "high", "low", "close", "volume"]
    full_df = validate_dataframe(full_df, required_cols)
    
    return full_df


def _compute_time_ranges(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))
    train_start_ts = df["datetime"].iloc[0]
    val_start_ts = df["datetime"].iloc[train_end] if train_end < n else df["datetime"].iloc[-1]
    test_start_ts = df["datetime"].iloc[val_end] if val_end < n else df["datetime"].iloc[-1]
    train_range = (train_start_ts.isoformat(), df["datetime"].iloc[train_end - 1].isoformat())
    val_range = (val_start_ts.isoformat(), df["datetime"].iloc[val_end - 1].isoformat() if val_end < n else df["datetime"].iloc[-1].isoformat())
    test_range = (test_start_ts.isoformat(), df["datetime"].iloc[-1].isoformat())
    return train_range, val_range, test_range


def process_pair(pair: str, args, batch_size: Optional[int] = None):
    years = args.years.split(",") if args.years else None
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = (ROOT / input_root).resolve()

    raw_df = _load_pair_data(pair, input_root, years)
    # ---------------------------------------------------------------------
    # 1️⃣  Ensure timestamps are UTC and deduplicate any overlapping rows.
    # ---------------------------------------------------------------------
    # HistData timestamps are in Central Time (America/Chicago). Converting
    # them to UTC provides a single source of truth across the pipeline.
    # ---------------------------------------------------------------------
    # 1️⃣  Ensure timestamps are UTC and deduplicate any overlapping rows.
    # ---------------------------------------------------------------------
    raw_df = convert_to_utc_and_dedup(raw_df, datetime_col="datetime")

    include_groups = None if args.feature_groups.lower() == "all" else [g.strip() for g in args.feature_groups.split(",") if g.strip()]
    exclude_groups = (
        [g.strip() for g in args.exclude_feature_groups.split(",") if g.strip()]
        if args.exclude_feature_groups
        else None
    )
    feature_cfg = FeatureConfig(
        sma_windows=[int(x) for x in args.sma_windows.split(",") if x.strip()],
        ema_windows=[int(x) for x in args.ema_windows.split(",") if x.strip()],
        rsi_window=args.rsi_window,
        bollinger_window=args.bollinger_window,
        bollinger_num_std=args.bollinger_num_std,
        atr_window=args.atr_window,
        short_vol_window=args.short_vol_window,
        long_vol_window=args.long_vol_window,
        spread_windows=[int(x) for x in args.spread_windows.split(",") if x.strip()],
        imbalance_smoothing=args.imbalance_smoothing,
        include_groups=include_groups,
        exclude_groups=exclude_groups,
    )
    df_for_features = raw_df
    if args.intrinsic_time:
        df_for_features = build_intrinsic_time_bars(
            raw_df,
            up_threshold=args.dc_threshold_up,
            down_threshold=args.dc_threshold_down or args.dc_threshold_up,
        )
    logger = get_logger(__name__)
    logger.info(
        "[intrinsic] reduced %d -> %d bars using DC thresholds up=%s, down=%s",
        len(raw_df),
        len(df_for_features),
        args.dc_threshold_up,
        args.dc_threshold_down or args.dc_threshold_up,
    )

    # ---------------------------------------------------------------------
    # 2️⃣  Cache heavy feature computation to avoid re‑running on every call.
    # ---------------------------------------------------------------------
    cache_dir = ROOT / "output_central" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Simple hash based on raw data shape and feature config parameters.

    # ---------------------------------------------------------------------
    # 3️⃣  Compute a robust cache key that includes the *content* of the raw
    #      dataframe.  This avoids stale caches when the underlying price data
    #      changes but the shape stays identical.
    # ---------------------------------------------------------------------
    import hashlib

    # Hash the raw data values (including index) – ``hash_pandas_object`` returns a
    # Series of uint64 hashes; converting to bytes yields a deterministic digest.
    raw_hash_bytes = pd.util.hash_pandas_object(raw_df, index=True).values.tobytes()
    hash_input = (
            raw_hash_bytes
            + str(feature_cfg).encode()
            + (b"intrinsic" if args.intrinsic_time else b"regular")
    )
    cache_hash = hashlib.sha256(hash_input).hexdigest()[:12]
    cache_path = cache_dir / f"{pair}_features_{cache_hash}.feather"
    if cache_path.exists():
        feature_df = pd.read_feather(cache_path)
        logger.info("[cache] loaded pre‑computed features from %s", cache_path)
    else:
        feature_df = build_feature_frame(df_for_features, config=feature_cfg)
        feature_df.to_feather(cache_path)
        logger.info("[cache] saved computed features to %s", cache_path)
    
    # Optional: Attach sentiment features if GDELT data available
    if args.include_sentiment:
        try:
            logger.info("[sentiment] attempting to attach news sentiment features...")
            feature_df = attach_sentiment_features(feature_df, pair=args.pairs)
            logger.info("[sentiment] successfully added sentiment columns to feature frame")
        except Exception as e:
            logger.warning(f"[sentiment] failed to attach sentiment features: {e}. Continuing without sentiment.")
    
    feature_df["datetime"] = pd.to_datetime(feature_df["datetime"])

    train_range, val_range, test_range = _compute_time_ranges(
        feature_df, args.train_ratio, args.val_ratio
    )

    feature_cols = [c for c in feature_df.columns if c not in {"datetime", "source_file"}]

    data_cfg = DataConfig(
        csv_path="",
        datetime_column="datetime",
        feature_columns=feature_cols,
        target_type=args.target_type,
        t_in=args.t_in,
        t_out=args.t_out,
        lookahead_window=args.lookahead_window,
        top_k_predictions=args.top_k,
        predict_sell_now=args.predict_sell_now,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        flat_threshold=args.flat_threshold,
    )

    # ---------------------------------------------------------------------
    # 5️⃣  Create datasets and dataloaders using the unified BaseDataAgent.
    # ---------------------------------------------------------------------
    from data.agents.single_task_agent import SingleTaskDataAgent
    from data.agents.base_agent import BaseDataAgent as DataAgent

    agent = SingleTaskDataAgent(data_cfg)
    datasets = agent.build_datasets(feature_df)
    effective_batch_size = batch_size if batch_size is not None else getattr(args, "batch_size", 64)
    loaders = DataAgent.build_dataloaders(datasets, batch_size=effective_batch_size)

    logger.info("Pair: %s", pair)
    logger.info("Rows after features: %s", f"{len(feature_df):,}")
    logger.info("Features: %s -> %s", len(feature_cols), feature_cols)
    for split, ds in datasets.items():
        loader = loaders[split]
        logger.info("%s: %s windows (batch_size=%s)", split, f"{len(ds):,}", loader.batch_size)
    logger.info("Dataloaders ready (train/val/test).")
    return pair, loaders


def main():
    # Ensure reproducibility across runs.
    set_seed()
    args = parse_args()
    pairs = [p.strip().lower() for p in args.pairs.split(",") if p.strip()]
    results = {}
    for pair in pairs:
        try:
            _, loaders = process_pair(pair, args)
            results[pair] = loaders
        except Exception as e:
            logger.error("[error] Failed to process %s: %s", pair, e)
    return results


if __name__ == "__main__":
    main()
