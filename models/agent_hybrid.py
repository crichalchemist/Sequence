import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import ModelConfig


class TemporalAttention(nn.Module):
    """Single‑head additive attention.

    Used when ``ModelConfig.use_multihead_attention`` is ``False`` (default).
    """

    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        h = torch.tanh(self.proj(x))
        scores = self.score(h).squeeze(-1)  # [B, T]
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return context, weights


class MultiHeadTemporalAttention(nn.Module):
    """Multi‑head additive attention (optional).

    Mirrors the API of ``TemporalAttention`` but splits the representation
    into ``n_heads`` sub‑spaces and concatenates the resulting contexts.
    """

    def __init__(self, input_dim: int, attention_dim: int, n_heads: int = 4):
        super().__init__()
        assert input_dim % n_heads == 0, "input_dim must be divisible by n_heads"
        self.n_heads = n_heads
        head_dim = input_dim // n_heads
        self.head_dim = head_dim
        self.proj = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, n_heads)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        B, T, D = x.shape
        h = torch.tanh(self.proj(x))  # [B, T, attention_dim]
        # Produce a score per head.
        scores = self.score(h)  # [B, T, n_heads]
        scores = scores.permute(0, 2, 1)  # [B, n_heads, T]
        weights = F.softmax(scores, dim=2)  # per‑head softmax over time
        # Split x into heads.
        x_heads = x.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, n_heads, T, head_dim]
        # Weighted sum per head.
        context = torch.einsum("bnht,bnhtd->bnhd", weights, x_heads)  # [B, n_heads, 1, head_dim]
        context = context.squeeze(2)  # [B, n_heads, head_dim]
        # Concatenate heads back to [B, D]
        context = context.reshape(B, D)
        # Return combined weights (average across heads for logging)
        avg_weights = weights.mean(dim=1)  # [B, T]
        return context, avg_weights


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
        # Choose attention variant based on config.
        if cfg.use_multihead_attention:
            self.attention = MultiHeadTemporalAttention(self.output_dim, cfg.attention_dim)
        else:
            self.attention = TemporalAttention(self.output_dim, cfg.attention_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, F]
        lstm_out, _ = self.lstm(x)  # [B, T, H_lstm * (1 or 2)]

        cnn_in = x.permute(0, 2, 1)  # [B, F, T]
        cnn_features = F.relu(self.cnn(cnn_in)).permute(0, 2, 1)  # [B, T, H_cnn]

        combined = torch.cat([lstm_out, cnn_features], dim=-1)  # [B, T, H_lstm + H_cnn]

        context, attn_weights = self.attention(combined)
        context = self.dropout(context)
        return context, attn_weights


class DignityModel(nn.Module):
    """
    CNN + BiLSTM + temporal attention Dignity model with multi-head outputs.
    """

    def __init__(self, cfg: ModelConfig, task_type: str = "classification"):
        super().__init__()
        self.cfg = cfg
        self.task_type = task_type

        self.price_encoder = PriceSequenceEncoder(cfg)

        attn_input_dim = self.price_encoder.output_dim
        self.head_direction = nn.Linear(attn_input_dim, cfg.num_dir_classes)
        self.head_return = nn.Linear(attn_input_dim, cfg.return_dim)
        self.head_volatility = nn.Linear(attn_input_dim, cfg.num_volatility_classes)
        self.head_max_return = nn.Linear(attn_input_dim, 1)
        self.head_topk_returns = nn.Linear(attn_input_dim, cfg.top_k_predictions)
        self.head_topk_prices = nn.Linear(attn_input_dim, cfg.top_k_predictions)
        self.head_sell_now = nn.Linear(attn_input_dim, 1) if cfg.predict_sell_now else None

    def forward(self, x: torch.Tensor):
        context, attn_weights = self.price_encoder(x)

        direction_logits = self.head_direction(context)
        return_pred = self.head_return(context)
        primary_output = direction_logits if self.task_type == "classification" else return_pred
        outputs = {
            "primary": primary_output,
            "direction_logits": direction_logits,
            "return": return_pred,
            "volatility_logits": self.head_volatility(context),
            "max_return": self.head_max_return(context),
            "topk_returns": self.head_topk_returns(context),
            "topk_prices": self.head_topk_prices(context),
        }
        if self.head_sell_now is not None:
            outputs["sell_now"] = self.head_sell_now(context)
        return outputs, attn_weights


def build_model(
    cfg: ModelConfig,
    task_type: str = "classification",
    num_dir_classes: int | None = None,
    num_volatility_classes: int | None = None,
    return_dim: int | None = None,
) -> DignityModel:
    if num_dir_classes is not None:
        cfg.num_dir_classes = num_dir_classes
    if num_volatility_classes is not None:
        cfg.num_volatility_classes = num_volatility_classes
    if return_dim is not None:
        cfg.return_dim = return_dim
    return DignityModel(cfg, task_type=task_type)


# Backward compatibility for existing checkpoints/imports.
HybridCNNLSTMAttention = DignityModel
