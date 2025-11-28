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

from config.config import DataConfig
from data.agent_data import DataAgent
from features.agent_features import build_feature_frame
from features.intrinsic_time import build_intrinsic_time_bars


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare datasets from Central-time HistData zips.")
    parser.add_argument("--pairs", default="gbpusd", help="Comma-separated pair codes (e.g., gbpusd,eurusd)")
    parser.add_argument("--input-root", default="output_central", help="Root directory containing pair subfolders with zips")
    parser.add_argument("--years", default=None, help="Comma-separated list of years to include (e.g., 2018,2019). Default: all")
    parser.add_argument("--t-in", type=int, default=120, help="Lookback window length")
    parser.add_argument("--t-out", type=int, default=10, help="Forecast horizon in minutes")
    parser.add_argument("--target-type", choices=["classification", "regression"], default="classification")
    parser.add_argument("--flat-threshold", type=float, default=0.0001, help="Flat class threshold for log returns")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train fraction (time-ordered)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation fraction (time-ordered)")
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
            print(f"[warn] failed to load CSV {cpath}: {exc}")
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
    df_for_features = raw_df
    if args.intrinsic_time:
        df_for_features = build_intrinsic_time_bars(
            raw_df,
            up_threshold=args.dc_threshold_up,
            down_threshold=args.dc_threshold_down,
        )
        print(
            f"[intrinsic] reduced {len(raw_df):,} -> {len(df_for_features):,} bars using "
            f"DC thresholds up={args.dc_threshold_up}, down={args.dc_threshold_down or args.dc_threshold_up}"
        )

    feature_df = build_feature_frame(df_for_features)
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
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        flat_threshold=args.flat_threshold,
    )

    agent = DataAgent(data_cfg)
    datasets = agent.build_datasets(feature_df)
    loaders = DataAgent.build_dataloaders(datasets, batch_size=64)

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
