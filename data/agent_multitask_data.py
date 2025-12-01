from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config.config import MultiTaskDataConfig
from data.agents.base_agent import _label_from_return, _select_range


@dataclass
class MultiTaskNormalizationStats:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


class MultiTaskSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: Dict[str, np.ndarray]):
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.targets = {
            k: torch.as_tensor(v, dtype=(torch.long if "class" in k else torch.float32)) for k, v in targets.items()
        }

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        x = self.sequences[idx]
        y = {k: v[idx] for k, v in self.targets.items()}
        return x, y


class MultiTaskDataAgent:
    """
    Prepares normalized sliding windows and multi-task labels for train/val/test splits.
    """

    def __init__(self, cfg: MultiTaskDataConfig):
        self.cfg = cfg
        self.norm_stats: Optional[MultiTaskNormalizationStats] = None

    def split_dataframe(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        time_col = self.cfg.datetime_column
        df = df.sort_values(time_col).reset_index(drop=True)
        return {
            "train": _select_range(df, time_col, self.cfg.train_range),
            "val": _select_range(df, time_col, self.cfg.val_range),
            "test": _select_range(df, time_col, self.cfg.test_range),
        }

    def fit_normalization(self, train_df: pd.DataFrame, feature_cols: List[str]) -> MultiTaskNormalizationStats:
        data = train_df[feature_cols].to_numpy(dtype=np.float32)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        self.norm_stats = MultiTaskNormalizationStats(mean=mean, std=std)
        return self.norm_stats

    def _build_windows(
        self, df: pd.DataFrame, feature_cols: List[str], norm_stats: MultiTaskNormalizationStats
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        t_in, t_out = self.cfg.t_in, self.cfg.t_out
        lookahead = self.cfg.lookahead_window or t_out
        top_k = max(1, self.cfg.top_k_predictions)
        features = norm_stats.apply(df[feature_cols].to_numpy(dtype=np.float32))

        log_close = np.log(df["close"].to_numpy())
        log_returns = np.diff(log_close, prepend=np.nan)

        sequences: List[np.ndarray] = []
        targets_dir_cls: List[int] = []
        targets_ret_reg: List[float] = []
        targets_next_close: List[float] = []
        targets_vol_cls: List[int] = []
        targets_max_return: List[float] = []
        targets_topk_returns: List[np.ndarray] = []
        targets_topk_prices: List[np.ndarray] = []
        targets_sell_now: List[int] = []
        targets_trend_cls: List[int] = []
        targets_vol_regime_cls: List[int] = []
        targets_candle_cls: List[int] = []

        last_idx = len(df) - max(t_out, lookahead)
        for idx in range(t_in - 1, last_idx):
            start_seq = idx - t_in + 1
            end_seq = idx + 1
            seq = features[start_seq:end_seq]

            future_close = df["close"].iloc[idx + t_out]
            future_log_ret = log_close[idx + t_out] - log_close[idx]
            if not np.isfinite(future_log_ret):
                continue

            direction_label = int(_label_from_return(future_log_ret, self.cfg.flat_threshold))
            return_target = float(future_log_ret)
            next_close_target = float(future_close)

            future_returns = log_close[idx + 1 : idx + lookahead + 1] - log_close[idx]
            if len(future_returns) < lookahead or np.isnan(future_returns).any():
                continue
            max_future_return = float(np.max(future_returns))
            sorted_returns = np.sort(future_returns)[::-1]
            topk_returns = sorted_returns[:top_k]
            if len(topk_returns) < top_k:
                topk_returns = np.pad(topk_returns, (0, top_k - len(topk_returns)), constant_values=sorted_returns[-1])
            topk_prices = np.exp(topk_returns) * df["close"].iloc[idx]

            past_ret_window = log_returns[idx - t_out + 1 : idx + 1]
            future_ret_window = log_returns[idx + 1 : idx + t_out + 1]
            if len(past_ret_window) < t_out or len(future_ret_window) < t_out:
                continue
            if np.isnan(past_ret_window).any() or np.isnan(future_ret_window).any():
                continue
            past_vol = float(np.std(past_ret_window))
            future_vol = float(np.std(future_ret_window))
            vol_label = 1 if (future_vol - past_vol) > self.cfg.vol_min_change else 0

            trend_avg_return = float(np.mean(future_ret_window))
            trend_label = int(_label_from_return(trend_avg_return, self.cfg.flat_threshold))

            vol_delta = future_vol - past_vol
            if vol_delta > self.cfg.vol_min_change:
                vol_regime_label = 2
            elif vol_delta < -self.cfg.vol_min_change:
                vol_regime_label = 0
            else:
                vol_regime_label = 1

            candle_row = df.iloc[idx]
            open_price = float(candle_row["open"])
            high_price = float(candle_row["high"])
            low_price = float(candle_row["low"])
            close_price = float(candle_row["close"])
            range_size = max(high_price - low_price, 1e-8)
            body = close_price - open_price
            body_ratio = abs(body) / range_size
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price

            if body_ratio < 0.1:
                candle_label = 0  # doji/indecision
            elif body > 0 and body_ratio > 0.55:
                candle_label = 1  # strong bullish body
            elif body < 0 and body_ratio > 0.55:
                candle_label = 2  # strong bearish body
            elif lower_wick / range_size > 0.35:
                candle_label = 3  # hammer/long lower wick
            else:
                candle_label = 3

            sequences.append(seq)
            targets_dir_cls.append(direction_label)
            targets_ret_reg.append(return_target)
            targets_next_close.append(next_close_target)
            targets_vol_cls.append(vol_label)
            targets_max_return.append(max_future_return)
            targets_topk_returns.append(topk_returns)
            targets_topk_prices.append(topk_prices)
            if self.cfg.predict_sell_now:
                targets_sell_now.append(int(max_future_return <= 0.0))
            targets_trend_cls.append(trend_label)
            targets_vol_regime_cls.append(vol_regime_label)
            targets_candle_cls.append(candle_label)

        if not sequences:
            raise ValueError("No sequences created; check t_in/t_out and data length.")

        targets = {
            "direction_class": np.array(targets_dir_cls),
            "return_reg": np.array(targets_ret_reg),
            "next_close_reg": np.array(targets_next_close),
            "vol_class": np.array(targets_vol_cls),
            "max_return": np.array(targets_max_return),
            "topk_returns": np.stack(targets_topk_returns),
            "topk_prices": np.stack(targets_topk_prices),
            "trend_class": np.array(targets_trend_cls),
            "vol_regime_class": np.array(targets_vol_regime_cls),
            "candle_class": np.array(targets_candle_cls),
        }
        if self.cfg.predict_sell_now:
            targets["sell_now"] = np.array(targets_sell_now)
        return np.stack(sequences), targets

    def build_datasets(self, df: pd.DataFrame) -> Dict[str, MultiTaskSequenceDataset]:
        feature_cols = self.cfg.feature_columns or [
            col for col in df.columns if col not in {self.cfg.datetime_column, "target", "label"}
        ]

        splits = self.split_dataframe(df)
        train_df = splits["train"]
        self.fit_normalization(train_df, feature_cols)
        assert self.norm_stats is not None

        datasets: Dict[str, MultiTaskSequenceDataset] = {}
        for split_name, split_df in splits.items():
            sequences, targets = self._build_windows(split_df, feature_cols, self.norm_stats)
            datasets[split_name] = MultiTaskSequenceDataset(sequences, targets)
        return datasets

    @staticmethod
    def build_dataloaders(
        datasets: Dict[str, MultiTaskSequenceDataset], batch_size: int, num_workers: int = 0
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
