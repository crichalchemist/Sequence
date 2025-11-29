"""Regime-aware hybrid model combining price and GDELT encoders."""
"""Regime-aware hybrid encoder/decoder stack for price + GDELT fusion."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import ModelConfig
from models.agent_hybrid import TemporalAttention
from models.regime_encoder import RegimeEncoder


class PriceSequenceEncoder(nn.Module):
    """CNN + (bi)LSTM + temporal attention encoder returning a single embedding."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        padding = cfg.cnn_kernel_size // 2
        self.cnn = nn.Conv1d(
            in_channels=cfg.num_features,
            out_channels=cfg.cnn_num_filters,
            kernel_size=cfg.cnn_kernel_size,
            padding=padding,
        )

        self.lstm = nn.LSTM(
            input_size=cfg.num_features,
            hidden_size=cfg.hidden_size_lstm,
            num_layers=cfg.num_layers_lstm,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )

        lstm_factor = 2 if cfg.bidirectional else 1
        self.output_dim = lstm_factor * cfg.hidden_size_lstm + cfg.cnn_num_filters
        self.attention = TemporalAttention(self.output_dim, cfg.attention_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, F]
        lstm_out, _ = self.lstm(x)
        cnn_in = x.permute(0, 2, 1)
        cnn_features = F.relu(self.cnn(cnn_in)).permute(0, 2, 1)
        combined = torch.cat([lstm_out, cnn_features], dim=-1)
        context, attn_weights = self.attention(combined)
        context = self.dropout(context)
        return context, attn_weights


class RegimeAwareHybrid(nn.Module):
    """Two-stream model that fuses price and regime embeddings."""

    def __init__(
        self,
        cfg: ModelConfig,
        shared_dim: int = 128,
        regime_input_dim: int = 18,
        regime_hidden_dim: int = 32,
        regime_emb_dim: int = 16,
        regime_classes: int = 3,
        vol_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.price_encoder = PriceSequenceEncoder(cfg)
        self.regime_encoder = RegimeEncoder(
            input_dim=regime_input_dim, hidden_dim=regime_hidden_dim, emb_dim=regime_emb_dim
        )

        fused_dim = self.price_encoder.output_dim + regime_emb_dim
        self.shared = nn.Linear(fused_dim, shared_dim)

        self.direction_head = nn.Linear(shared_dim, cfg.num_dir_classes)
        self.return_head = nn.Linear(shared_dim, cfg.return_dim)
        self.regime_class_head = nn.Linear(shared_dim, regime_classes)
        self.volatility_head = nn.Linear(shared_dim, vol_classes or cfg.num_volatility_classes)
        self.trade_suppress_head = nn.Linear(shared_dim, 1)

    def forward(self, price_seq: torch.Tensor, regime_vec: torch.Tensor) -> tuple[dict, torch.Tensor]:
        price_emb, attn_weights = self.price_encoder(price_seq)
        regime_emb = self.regime_encoder(regime_vec)

        fused = torch.cat([price_emb, regime_emb], dim=-1)
        h = torch.relu(self.shared(fused))

        outputs = {
            "direction_logits": self.direction_head(h),
            "return_pred": self.return_head(h),
            "regime_logits": self.regime_class_head(h),
            "vol_logits": self.volatility_head(h),
            "suppress_logit": self.trade_suppress_head(h),
        }
        return outputs, attn_weights
