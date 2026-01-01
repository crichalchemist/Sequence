"""Regime-aware hybrid model combining price and GDELT encoders."""
from __future__ import annotations

import torch
import torch.nn as nn
from config.config import ModelConfig

from models.agent_hybrid import PriceSequenceEncoder
from models.regime_encoder import RegimeEncoder


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
