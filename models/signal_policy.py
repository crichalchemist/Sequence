import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import PolicyConfig, SignalModelConfig
from models.agent_hybrid import TemporalAttention


class SignalBackbone(nn.Module):
    """CNN + LSTM + attention encoder reused by both signal heads and the policy."""

    def __init__(self, cfg: SignalModelConfig):
        super().__init__()
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
        self.output_dim = attn_input_dim

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)  # [B, T, H_lstm]
        cnn_in = x.permute(0, 2, 1)
        cnn_features = F.relu(self.cnn(cnn_in)).permute(0, 2, 1)
        combined = torch.cat([lstm_out, cnn_features], dim=-1)
        context, attn_weights = self.attention(combined)
        return context, attn_weights


class SignalModel(nn.Module):
    """Hybrid encoder with optional forecasting and direction heads."""

    def __init__(self, cfg: SignalModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = SignalBackbone(cfg)
        self.dropout = nn.Dropout(cfg.signal_dropout)
        self.head_direction = (
            nn.Linear(self.backbone.output_dim, cfg.num_classes or 3)
            if cfg.use_direction_head
            else None
        )
        self.head_forecast = (
            nn.Linear(self.backbone.output_dim, cfg.forecast_output_dim)
            if cfg.use_forecast_head
            else None
        )

    @property
    def signal_dim(self) -> int:
        return self.backbone.output_dim

    def forward(self, x: torch.Tensor):
        embedding, attn_weights = self.backbone(x)
        embedding = self.dropout(embedding)
        aux_outputs = {}
        if self.head_direction is not None:
            aux_outputs["direction_logits"] = self.head_direction(embedding)
        if self.head_forecast is not None:
            aux_outputs["forecast"] = self.head_forecast(embedding)
        return {"embedding": embedding, "aux": aux_outputs, "attn": attn_weights}


class ExecutionPolicy(nn.Module):
    """PPO/A3C style execution head that consumes signal embeddings."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        value_hidden = cfg.value_hidden_dim or cfg.hidden_dim
        self.policy_net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_actions),
        )
        self.value_net = nn.Sequential(
            nn.Linear(cfg.input_dim, value_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, signal_embedding: torch.Tensor, detach_signal: bool = False):
        if detach_signal:
            signal_embedding = signal_embedding.detach()
        policy_logits = self.policy_net(signal_embedding)
        value = self.value_net(signal_embedding).squeeze(-1)
        return policy_logits, value


class SignalPolicyAgent(nn.Module):
    """End-to-end module combining the signal model with the execution policy."""

    def __init__(self, signal_model: SignalModel, policy: ExecutionPolicy):
        super().__init__()
        self.signal_model = signal_model
        self.policy = policy

    def forward(self, x: torch.Tensor, detach_signal: bool = False):
        signal_out = self.signal_model(x)
        logits, value = self.policy(signal_out["embedding"], detach_signal=detach_signal)
        return {
            "policy_logits": logits,
            "value": value,
            "signal": signal_out,
        }

    @classmethod
    def load(
        cls,
        signal_cfg: SignalModelConfig,
        policy_cfg: PolicyConfig,
        signal_path: str,
        policy_path: str,
        device: torch.device,
    ) -> "SignalPolicyAgent":
        signal_model = SignalModel(signal_cfg).to(device)
        policy = ExecutionPolicy(policy_cfg).to(device)
        signal_state = torch.load(signal_path, map_location=device)
        policy_state = torch.load(policy_path, map_location=device)
        signal_model.load_state_dict(signal_state)
        policy.load_state_dict(policy_state)
        return cls(signal_model, policy)
