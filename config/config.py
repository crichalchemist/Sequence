from dataclasses import dataclass, field
from typing import List, Optional, Tuple


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
    bidirectional: bool = True
    num_dir_classes: Optional[int] = None
    return_dim: Optional[int] = None
    num_volatility_classes: int = 2

    def __post_init__(self) -> None:
        # Preserve backward compatibility with older configs that only specify
        # num_classes/output_dim while allowing explicit head dimensions.
        if self.num_dir_classes is None:
            self.num_dir_classes = self.num_classes or 3
        if self.return_dim is None:
            self.return_dim = self.output_dim


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
