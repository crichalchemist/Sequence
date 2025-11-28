import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import MultiTaskModelConfig
from models.agent_hybrid import TemporalAttention


class MultiHeadHybrid(nn.Module):
    """
    Shared CNN + LSTM + temporal attention encoder with seven heads:
    1) direction classification
    2) return regression
    3) next close regression
    4) volatility direction classification
    5) trend classification
    6) volatility regime classification
    7) candle-pattern classification
    """

    def __init__(self, cfg: MultiTaskModelConfig):
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
        )

        attn_input_dim = cfg.hidden_size_lstm + cfg.cnn_num_filters
        self.attention = TemporalAttention(attn_input_dim, cfg.attention_dim)

        self.dropout = nn.Dropout(cfg.dropout)
        self.head_direction = nn.Linear(attn_input_dim, cfg.num_dir_classes)
        self.head_return = nn.Linear(attn_input_dim, 1)
        self.head_next_close = nn.Linear(attn_input_dim, 1)
        self.head_volatility = nn.Linear(attn_input_dim, cfg.num_vol_classes)
        self.head_trend = nn.Linear(attn_input_dim, cfg.num_trend_classes)
        self.head_vol_regime = nn.Linear(attn_input_dim, cfg.num_vol_regime_classes)
        self.head_candle = nn.Linear(attn_input_dim, cfg.num_candle_classes)

    def forward(self, x: torch.Tensor):
        # x: [B, T, F]
        lstm_out, _ = self.lstm(x)  # [B, T, H_lstm]

        cnn_in = x.permute(0, 2, 1)  # [B, F, T]
        cnn_features = F.relu(self.cnn(cnn_in)).permute(0, 2, 1)  # [B, T, H_cnn]

        combined = torch.cat([lstm_out, cnn_features], dim=-1)  # [B, T, H_lstm + H_cnn]
        context, attn_weights = self.attention(combined)
        context = self.dropout(context)

        outputs = {
            "direction_logits": self.head_direction(context),
            "return": self.head_return(context),
            "next_close": self.head_next_close(context),
            "volatility_logits": self.head_volatility(context),
            "trend_logits": self.head_trend(context),
            "vol_regime_logits": self.head_vol_regime(context),
            "candle_pattern_logits": self.head_candle(context),
        }
        return outputs, attn_weights


def build_multitask_model(cfg: MultiTaskModelConfig) -> MultiHeadHybrid:
    return MultiHeadHybrid(cfg)
