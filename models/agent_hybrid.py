import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import ModelConfig


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        h = torch.tanh(self.proj(x))
        scores = self.score(h).squeeze(-1)  # [B, T]
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return context, weights


class HybridCNNLSTMAttention(nn.Module):
    """
    CNN + BiLSTM + temporal attention hybrid model with multi-head outputs.
    """

    def __init__(self, cfg: ModelConfig, task_type: str = "classification"):
        super().__init__()
        self.cfg = cfg
        self.task_type = task_type

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
        attn_input_dim = lstm_factor * cfg.hidden_size_lstm + cfg.cnn_num_filters
        self.attention = TemporalAttention(attn_input_dim, cfg.attention_dim)

        self.dropout = nn.Dropout(cfg.dropout)
        self.head_direction = nn.Linear(attn_input_dim, cfg.num_dir_classes)
        self.head_return = nn.Linear(attn_input_dim, cfg.return_dim)
        self.head_volatility = nn.Linear(attn_input_dim, cfg.num_volatility_classes)

    def forward(self, x: torch.Tensor):
        # x: [B, T, F]
        lstm_out, _ = self.lstm(x)  # [B, T, H_lstm * (1 or 2)]

        cnn_in = x.permute(0, 2, 1)  # [B, F, T]
        cnn_features = F.relu(self.cnn(cnn_in)).permute(0, 2, 1)  # [B, T, H_cnn]

        combined = torch.cat([lstm_out, cnn_features], dim=-1)  # [B, T, H_lstm + H_cnn]

        context, attn_weights = self.attention(combined)
        context = self.dropout(context)
        outputs = {
            "direction_logits": self.head_direction(context),
            "return": self.head_return(context),
            "volatility_logits": self.head_volatility(context),
        }
        return outputs, attn_weights


def build_model(
    cfg: ModelConfig,
    task_type: str = "classification",
    num_dir_classes: int | None = None,
    num_volatility_classes: int | None = None,
    return_dim: int | None = None,
) -> HybridCNNLSTMAttention:
    if num_dir_classes is not None:
        cfg.num_dir_classes = num_dir_classes
    if num_volatility_classes is not None:
        cfg.num_volatility_classes = num_volatility_classes
    if return_dim is not None:
        cfg.return_dim = return_dim
    return HybridCNNLSTMAttention(cfg, task_type=task_type)
