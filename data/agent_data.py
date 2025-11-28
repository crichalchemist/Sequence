from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config.config import DataConfig


def _select_range(df: pd.DataFrame, time_col: str, date_range: Optional[Tuple[str, str]]) -> pd.DataFrame:
    if date_range is None:
        return df
    start, end = date_range
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    return df.loc[(df[time_col] >= start) & (df[time_col] <= end)]


def _compute_future_log_return(close: pd.Series, horizon: int) -> pd.Series:
    future = close.shift(-horizon)
    return np.log(future / close)


def _label_from_return(log_ret: float, flat_threshold: float) -> int:
    if log_ret > flat_threshold:
        return 2  # up
    if log_ret < -flat_threshold:
        return 0  # down
    return 1  # flat


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        targets: Dict[str, np.ndarray],
        target_type: str,
    ):
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.targets = {
            "primary": torch.as_tensor(
                targets["primary"], dtype=(torch.long if target_type == "classification" else torch.float32)
            ),
            "max_return": torch.as_tensor(targets["max_return"], dtype=torch.float32),
            "topk_returns": torch.as_tensor(targets["topk_returns"], dtype=torch.float32),
            "topk_prices": torch.as_tensor(targets["topk_prices"], dtype=torch.float32),
        }
        if "sell_now" in targets:
            self.targets["sell_now"] = torch.as_tensor(targets["sell_now"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        x = self.sequences[idx]
        y = {k: v[idx] for k, v in self.targets.items()}
        return x, y


class DataAgent:
    """
    Prepares normalized sliding windows and labels for train/val/test splits.
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        if self.cfg.target_type not in {"classification", "regression"}:
            raise ValueError("target_type must be 'classification' or 'regression'")
        self.norm_stats: Optional[NormalizationStats] = None

    def split_dataframe(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        time_col = self.cfg.datetime_column
        df = df.sort_values(time_col).reset_index(drop=True)
        return {
            "train": _select_range(df, time_col, self.cfg.train_range),
            "val": _select_range(df, time_col, self.cfg.val_range),
            "test": _select_range(df, time_col, self.cfg.test_range),
        }

    def fit_normalization(self, train_df: pd.DataFrame, feature_cols: List[str]) -> NormalizationStats:
        data = train_df[feature_cols].to_numpy(dtype=np.float32)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        self.norm_stats = NormalizationStats(mean=mean, std=std)
        return self.norm_stats

    def _build_windows(
        self, df: pd.DataFrame, feature_cols: List[str], norm_stats: NormalizationStats
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        t_in, t_out = self.cfg.t_in, self.cfg.t_out
        lookahead = self.cfg.lookahead_window or t_out
        top_k = max(1, self.cfg.top_k_predictions)
        features = norm_stats.apply(df[feature_cols].to_numpy(dtype=np.float32))
        future_log_ret = _compute_future_log_return(df["close"], t_out).to_numpy()

        sequences: List[np.ndarray] = []
        primary_targets: List[float] = []
        max_return_targets: List[float] = []
        topk_returns_targets: List[np.ndarray] = []
        topk_prices_targets: List[np.ndarray] = []
        sell_now_targets: List[int] = []

        last_idx = len(df) - max(t_out, lookahead)
        log_close = np.log(df["close"].to_numpy())
        for idx in range(t_in - 1, last_idx):
            start = idx - t_in + 1
            end = idx + 1
            seq = features[start:end]
            target_return = future_log_ret[idx]
            if not np.isfinite(target_return):
                continue
            if self.cfg.target_type == "classification":
                target = int(_label_from_return(target_return, self.cfg.flat_threshold))
            else:
                target = float(target_return)

            future_returns = log_close[idx + 1 : idx + lookahead + 1] - log_close[idx]
            if len(future_returns) < lookahead or np.isnan(future_returns).any():
                continue
            max_future_return = float(np.max(future_returns))
            sorted_returns = np.sort(future_returns)[::-1]
            topk_returns = sorted_returns[:top_k]
            if len(topk_returns) < top_k:
                topk_returns = np.pad(topk_returns, (0, top_k - len(topk_returns)), constant_values=sorted_returns[-1])
            topk_prices = np.exp(topk_returns) * df["close"].iloc[idx]

            sequences.append(seq)
            primary_targets.append(target)
            max_return_targets.append(max_future_return)
            topk_returns_targets.append(topk_returns)
            topk_prices_targets.append(topk_prices)
            if self.cfg.predict_sell_now:
                sell_now_targets.append(int(max_future_return <= 0.0))

        if not sequences:
            raise ValueError("No sequences created; check t_in/t_out and data length.")

        targets: Dict[str, np.ndarray] = {
            "primary": np.array(primary_targets),
            "max_return": np.array(max_return_targets),
            "topk_returns": np.stack(topk_returns_targets),
            "topk_prices": np.stack(topk_prices_targets),
        }
        if self.cfg.predict_sell_now:
            targets["sell_now"] = np.array(sell_now_targets)

        return np.stack(sequences), targets

    def build_datasets(self, df: pd.DataFrame) -> Dict[str, SequenceDataset]:
        feature_cols = self.cfg.feature_columns or [
            col for col in df.columns if col not in {self.cfg.datetime_column, "target", "label"}
        ]

        splits = self.split_dataframe(df)
        train_df = splits["train"]
        self.fit_normalization(train_df, feature_cols)
        assert self.norm_stats is not None

        datasets: Dict[str, SequenceDataset] = {}
        for split_name, split_df in splits.items():
            sequences, targets = self._build_windows(split_df, feature_cols, self.norm_stats)
            datasets[split_name] = SequenceDataset(sequences, targets, self.cfg.target_type)

        return datasets

    @staticmethod
    def build_dataloaders(
        datasets: Dict[str, SequenceDataset], batch_size: int, num_workers: int = 0
    ) -> Dict[str, torch.utils.data.DataLoader]:
        loaders = {
            name: torch.utils.data.DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=name == "train",
                num_workers=num_workers,
                drop_last=False,
            )
            for name, ds in datasets.items()
        }
        return loaders
