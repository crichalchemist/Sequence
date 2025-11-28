"""
Utility to load HistData CSVs (Central time), build features, and create train/val/test datasets
for the multi-head model.
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

from config.config import FeatureConfig, MultiTaskDataConfig
from data.agent_multitask_data import MultiTaskDataAgent
from features.agent_features import build_feature_frame
from features.agent_sentiment import aggregate_sentiment, attach_sentiment_features
from data.gdelt_ingest import load_gdelt_gkg


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare datasets for multi-task training.")
    parser.add_argument("--pairs", default="gbpusd", help="Comma-separated pair codes (e.g., gbpusd,eurusd)")
    parser.add_argument("--input-root", default="output_central", help="Root directory containing pair subfolders with zips")
    parser.add_argument("--years", default=None, help="Comma-separated list of years to include (e.g., 2018,2019). Default: all")
    parser.add_argument("--t-in", type=int, default=120, help="Lookback window length")
    parser.add_argument("--t-out", type=int, default=10, help="Forecast horizon in minutes")
    parser.add_argument("--flat-threshold", type=float, default=0.0001, help="Flat class threshold for log returns")
    parser.add_argument("--vol-min-change", type=float, default=0.0, help="Volatility delta threshold for vol direction label")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train fraction (time-ordered)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation fraction (time-ordered)")
    parser.add_argument("--gdelt-path", default=None, help="Optional path to a GDELT GKG file for sentiment features")
    parser.add_argument("--gdelt-tz", default="UTC", help="Timezone to convert GDELT timestamps into (default UTC)")
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
    return parser.parse_args()


def _load_pair_data(pair: str, input_root: Path, years: Optional[List[str]]) -> pd.DataFrame:
    pair_dir = input_root / pair
    if not pair_dir.exists():
        raise FileNotFoundError(f"No data folder for pair {pair} under {input_root}")

    zips = sorted(pair_dir.glob("*.zip"))
    if years:
        zips = [z for z in zips if any(y in z.name for y in years)]
    if not zips:
        raise FileNotFoundError(f"No zip files found for pair {pair} with years={years}")

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
    if not frames:
        raise RuntimeError("No CSV data loaded; check zip contents.")

    full_df = pd.concat(frames, ignore_index=True)
    full_df["datetime"] = pd.to_datetime(full_df["datetime"], format="%Y%m%d %H%M%S")
    full_df = full_df.sort_values("datetime").reset_index(drop=True)
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


def process_pair(pair: str, args):
    years = args.years.split(",") if args.years else None
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = (ROOT / input_root).resolve()

    raw_df = _load_pair_data(pair, input_root, years)
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
    feature_df = build_feature_frame(raw_df, config=feature_cfg)
    feature_df["datetime"] = pd.to_datetime(feature_df["datetime"])

    if args.gdelt_path:
        try:
            gdelt_df = load_gdelt_gkg(Path(args.gdelt_path), target_tz=args.gdelt_tz)
            sent_feats = aggregate_sentiment(gdelt_df, feature_df, time_col="datetime", score_col="sentiment_score")
            feature_df = attach_sentiment_features(feature_df, sent_feats)
            print(f"GDELT sentiment merged: {len(sent_feats.columns)} sentiment features")
        except Exception as exc:
            print(f"[warn] Failed to ingest GDELT sentiment: {exc}")

    train_range, val_range, test_range = _compute_time_ranges(
        feature_df, args.train_ratio, args.val_ratio
    )

    feature_cols = [c for c in feature_df.columns if c not in {"datetime", "source_file"}]

    data_cfg = MultiTaskDataConfig(
        csv_path="",
        datetime_column="datetime",
        feature_columns=feature_cols,
        t_in=args.t_in,
        t_out=args.t_out,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        flat_threshold=args.flat_threshold,
        vol_min_change=args.vol_min_change,
    )

    agent = MultiTaskDataAgent(data_cfg)
    datasets = agent.build_datasets(feature_df)
    loaders = MultiTaskDataAgent.build_dataloaders(datasets, batch_size=64)

    print(f"Pair: {pair}")
    print(f"Rows after features: {len(feature_df):,}")
    print(f"Features: {len(feature_cols)} -> {feature_cols}")
    for split, ds in datasets.items():
        print(f"{split}: {len(ds):,} windows")
    print("Dataloaders ready (train/val/test).")
    return pair, loaders


def main():
    args = parse_args()
    pairs = [p.strip().lower() for p in args.pairs.split(",") if p.strip()]
    results = {}
    for pair in pairs:
        try:
            _, loaders = process_pair(pair, args)
            results[pair] = loaders
        except Exception as e:
            print(f"[error] Failed to process {pair}: {e}")
    return results


if __name__ == "__main__":
    main()
