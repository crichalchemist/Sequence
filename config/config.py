from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from risk.risk_manager import RiskConfig


@dataclass
class FeatureConfig:
    """Configuration for feature engineering windows and toggles."""

    sma_windows: List[int] = field(default_factory=lambda: [10, 20, 50])
    ema_windows: List[int] = field(default_factory=lambda: [10, 20, 50])
    rsi_window: int = 14
    bollinger_window: int = 20
    bollinger_num_std: float = 2.0
    atr_window: int = 14
    short_vol_window: int = 10
    long_vol_window: int = 50
    spread_windows: List[int] = field(default_factory=lambda: [20])
    imbalance_smoothing: int = 5
    include_groups: Optional[List[str]] = None
    exclude_groups: Optional[List[str]] = None


@dataclass
class DataConfig:
    csv_path: str
    datetime_column: str = "datetime"
    feature_columns: Optional[List[str]] = None
    target_type: str = "classification"  # "classification" or "regression"
    t_in: int = 120
    t_out: int = 10
    lookahead_window: Optional[int] = None  # window for auxiliary targets (defaults to t_out)
    top_k_predictions: int = 3
    predict_sell_now: bool = False
    train_range: Optional[Tuple[str, str]] = None  # ISO date strings
    val_range: Optional[Tuple[str, str]] = None
    test_range: Optional[Tuple[str, str]] = None
    flat_threshold: float = 0.0001  # abs(log return) below this -> flat


@dataclass
class ModelConfig:
    num_features: int
    hidden_size_lstm: int = 64
    num_layers_lstm: int = 1
    cnn_num_filters: int = 32
    cnn_kernel_size: int = 3
    attention_dim: int = 64
    dropout: float = 0.1
    num_classes: Optional[int] = 3  # set to None for regression
    output_dim: int = 1  # used for regression
    lookahead_window: Optional[int] = None
    top_k_predictions: int = 3
    predict_sell_now: bool = False
    bidirectional: bool = True
    num_dir_classes: Optional[int] = None
    return_dim: Optional[int] = None
    num_volatility_classes: int = 2

    # New optional flag to enable multi‑head attention in the encoder.
    use_multihead_attention: bool = False

    def __post_init__(self) -> None:
        # Preserve backward compatibility with older configs that only specify
        # num_classes/output_dim while allowing explicit head dimensions.
        if self.num_dir_classes is None:
            self.num_dir_classes = self.num_classes or 3
        if self.return_dim is None:
            self.return_dim = self.output_dim


@dataclass
class SignalModelConfig(ModelConfig):
    """
    Configuration for the hybrid signal encoder that feeds the execution policy.
    """

    use_direction_head: bool = True
    use_forecast_head: bool = True
    forecast_output_dim: int = 1
    signal_dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"
    grad_clip: Optional[float] = 1.0
    log_every: int = 50
    checkpoint_path: str = "models/best_model.pt"
    risk: RiskConfig = field(default_factory=RiskConfig)
    max_return_weight: float = 1.0
    topk_return_weight: float = 1.0
    topk_price_weight: float = 1.0
    sell_now_weight: float = 1.0
    # Early‑stopping patience (epochs without improvement) and checkpoint retention.
    early_stop_patience: int = 3
    top_n_checkpoints: int = 3


@dataclass
class PolicyConfig:
    """
    Configuration for the PPO/A3C-style execution policy head.
    """

    input_dim: int
    hidden_dim: int = 128
    num_actions: int = 3
    value_hidden_dim: Optional[int] = None
    dropout: float = 0.1


@dataclass
class RLTrainingConfig:
    """
    Training configuration for policy optimization that consumes signal outputs.
    """

    epochs: int = 5
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gamma: float = 0.99
    grad_clip: Optional[float] = 1.0
    detach_signal: bool = True
    checkpoint_path: str = "models/best_policy.pt"


@dataclass
class MultiTaskModelConfig:
    """
    Configuration for multi-head model with shared encoder.
    """
    num_features: int
    hidden_size_lstm: int = 64
    num_layers_lstm: int = 1
    cnn_num_filters: int = 32
    cnn_kernel_size: int = 3
    attention_dim: int = 64
    dropout: float = 0.1
    num_dir_classes: int = 3
    num_vol_classes: int = 2
    lookahead_window: Optional[int] = None
    top_k_predictions: int = 3
    predict_sell_now: bool = False
    num_trend_classes: int = 3
    num_vol_regime_classes: int = 3
    num_candle_classes: int = 4


@dataclass
class MultiTaskDataConfig:
    """
    Data config for multi-target setup.
    """
    csv_path: str
    datetime_column: str = "datetime"
    feature_columns: Optional[List[str]] = None
    t_in: int = 120
    t_out: int = 10
    lookahead_window: Optional[int] = None
    top_k_predictions: int = 3
    predict_sell_now: bool = False
    train_range: Optional[Tuple[str, str]] = None
    val_range: Optional[Tuple[str, str]] = None
    test_range: Optional[Tuple[str, str]] = None
    flat_threshold: float = 0.0001  # for direction classes
    vol_min_change: float = 0.0     # minimal vol delta to call it "up"


@dataclass
class MultiTaskLossWeights:
    """
    Loss weights for each task.
    """
    direction_cls: float = 1.0
    return_reg: float = 1.0
    next_close_reg: float = 1.0
    vol_cls: float = 1.0
    max_return_reg: float = 1.0
    topk_return_reg: float = 1.0
    topk_price_reg: float = 1.0
    sell_now_cls: float = 1.0
    trend_cls: float = 1.0
    vol_regime_cls: float = 1.0
    candle_pattern_cls: float = 1.0


@dataclass
class ExportConfig:
    onnx_path: str = "models/hybrid_cnn_lstm_attention.onnx"
    opset_version: int = 17


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    export: ExportConfig = field(default_factory=ExportConfig)
