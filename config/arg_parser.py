"""
Shared argument parsing utilities for training, evaluation, and pipeline scripts.

This module provides reusable argument parser factories to avoid duplication
across different entry points.
"""

import argparse
from pathlib import Path

from config.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_WEIGHT_DECAY,
)
from features.constants import DEFAULT_DC_THRESHOLD


def add_data_preparation_args(parser: argparse.ArgumentParser) -> None:
    """Add common data preparation and preprocessing arguments."""
    parser.add_argument("--pairs", default="gbpusd", help="Comma-separated pair codes")
    parser.add_argument("--years", default=None, help="Comma-separated years to include (default: all available)")
    parser.add_argument("--input-root", default="output_central", help="Root containing Central-time zips")
    parser.add_argument("--t-in", type=int, default=120, help="Lookback window length")
    parser.add_argument("--t-out", type=int, default=10, help="Prediction horizon")
    parser.add_argument("--task-type", choices=["classification", "regression"], default="classification")
    parser.add_argument("--flat-threshold", type=float, default=0.0001)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)


def add_feature_engineering_args(parser: argparse.ArgumentParser) -> None:
    """Add common feature engineering arguments."""
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


def add_intrinsic_time_args(parser: argparse.ArgumentParser) -> None:
    """Add intrinsic time (directional-change) transformation arguments."""
    parser.add_argument(
        "--intrinsic-time",
        action="store_true",
        help="Convert minute bars to intrinsic-time bars via directional-change events.",
    )
    parser.add_argument(
        "--dc-threshold-up",
        type=float,
        default=DEFAULT_DC_THRESHOLD,
        help="Fractional increase needed to flag an upward directional change (e.g., 0.001=0.1%).",
    )
    parser.add_argument(
        "--dc-threshold-down",
        type=float,
        default=None,
        help="Fractional decrease needed to flag a downward directional change. Defaults to dc-threshold-up.",
    )


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add common training arguments."""
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--device", default="cuda")


def add_auxiliary_task_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for auxiliary prediction tasks."""
    parser.add_argument("--lookahead-window", type=int, default=None, help="Lookahead for auxiliary targets")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K future returns/prices predictions")
    parser.add_argument("--predict-sell-now", action="store_true", help="Enable sell-now auxiliary head")


def add_dataloader_args(parser: argparse.ArgumentParser) -> None:
    """Add PyTorch DataLoader optimization arguments."""
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader workers for parallel loading")
    parser.add_argument("--pin-memory", action="store_true", default=True, help="Pin DataLoader memory for GPU")
    parser.add_argument("--prefetch-factor", type=int, default=DEFAULT_PREFETCH_FACTOR, help="DataLoader prefetch factor")
