from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from risk.risk_manager import RiskConfig


@dataclass
class DataConfig:
    csv_path: str
    datetime_column: str = "datetime"
    feature_columns: Optional[List[str]] = None
    target_type: str = "classification"  # "classification" or "regression"
    t_in: int = 120
    t_out: int = 10
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
