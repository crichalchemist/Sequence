from dataclasses import dataclass, field

from risk.risk_manager import RiskConfig


@dataclass
class FeatureConfig:
    """Configuration for feature engineering windows and toggles."""

    sma_windows: list[int] = field(default_factory=lambda: [10, 20, 50])
    ema_windows: list[int] = field(default_factory=lambda: [10, 20, 50])
    rsi_window: int = 14
    bollinger_window: int = 20
    bollinger_num_std: float = 2.0
    atr_window: int = 14
    short_vol_window: int = 10
    long_vol_window: int = 50
    spread_windows: list[int] = field(default_factory=lambda: [20])
    imbalance_smoothing: int = 5
    microstructure_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    dc_threshold_up: float = 0.001  # Directional change threshold for upward moves (0.1%)
    dc_threshold_down: float | None = None  # Defaults to dc_threshold_up if not specified
    include_groups: list[str] | None = None
    exclude_groups: list[str] | None = None


@dataclass
class CogneeConfig:
    """Configuration for Cognee Cloud knowledge graph features."""

    enable_cognee: bool = False
    api_key: str | None = None
    dataset_name: str = "fx_trading"
    rebuild_graph: bool = False
    entity_mention_window_hours: int = 24
    event_proximity_window_hours: int = 48
    include_economic_indicators: bool = True
    include_price_narratives: bool = True
    feature_types: list[str] = field(default_factory=lambda: [
        "entity_mentions",
        "event_proximity",
        "causal_chains",
        "pattern_similarity"
    ])


@dataclass
class DataConfig:
    csv_path: str
    datetime_column: str = "datetime"
    feature_columns: list[str] | None = None
    target_type: str = "classification"  # "classification" or "regression"
    t_in: int = 120
    t_out: int = 10
    lookahead_window: int | None = None  # window for auxiliary targets (defaults to t_out)
    top_k_predictions: int = 3
    predict_sell_now: bool = False
    train_range: tuple[str, str] | None = None  # ISO date strings
    val_range: tuple[str, str] | None = None
    test_range: tuple[str, str] | None = None
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
    num_classes: int | None = 3  # set to None for regression
    output_dim: int = 1  # used for regression
    lookahead_window: int | None = None
    top_k_predictions: int = 3
    predict_sell_now: bool = False
    bidirectional: bool = True
    num_dir_classes: int | None = None
    return_dim: int | None = None
    num_volatility_classes: int = 2

    # Attention optimization flags
    use_multihead_attention: bool = False
    use_optimized_attention: bool = True  # Enable memory-optimized attention
    max_seq_length: int = 1024  # Max sequence length for attention optimization
    use_chunking: bool = True  # Use chunking for long sequences
    chunk_size: int = 512  # Chunk size for memory efficiency
    use_adaptive_attention: bool = True  # Use adaptive attention mechanism
    complexity_threshold: float = 0.5  # Threshold for adaptive attention
    n_attention_heads: int = 4  # Number of attention heads
    sliding_window_size: int = 512  # Window size for sliding window attention
    sliding_step_size: int = 256  # Step size for sliding window attention

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
    grad_clip: float | None = 1.0
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

    # Advanced training optimizations
    use_amp: bool = False  # Enable Automatic Mixed Precision
    fp16: bool = False  # Use FP16 instead of FP32
    grad_scaler: str | None = None  # 'grad_scaler' or None for AMP

    # Asynchronous checkpoint saving
    async_checkpoint: bool = False
    checkpoint_workers: int = 2
    checkpoint_queue_size: int = 10

    # OpenTelemetry tracing configuration
    enable_tracing: bool = True  # Enable/disable tracing
    tracing_service_name: str = "sequence-training"
    tracing_otlp_endpoint: str = "http://localhost:4318"
    tracing_environment: str = "development"


@dataclass
class PolicyConfig:
    """
    Configuration for the PPO/A3C-style execution policy head.
    """

    input_dim: int
    hidden_dim: int = 128
    num_actions: int = 3
    value_hidden_dim: int | None = None
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
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation
    grad_clip: float | None = 1.0
    detach_signal: bool = True
    checkpoint_path: str = "models/best_policy.pt"

    # PPO-specific parameters
    use_ppo: bool = True  # Use PPO instead of vanilla policy gradient
    clip_range: float = 0.2  # PPO clipping parameter epsilon
    ppo_epochs: int = 4  # Number of optimization epochs per batch
    max_grad_norm: float = 0.5  # Maximum gradient norm for clipping
    target_kl: float | None = 0.01  # Early stopping if KL divergence exceeds this


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
    lookahead_window: int | None = None
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
    feature_columns: list[str] | None = None
    t_in: int = 120
    t_out: int = 10
    lookahead_window: int | None = None
    top_k_predictions: int = 3
    predict_sell_now: bool = False
    train_range: tuple[str, str] | None = None
    val_range: tuple[str, str] | None = None
    test_range: tuple[str, str] | None = None
    flat_threshold: float = 0.0001
    vol_min_change: float = 0.0  # minimal vol delta to call it "up"


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


# New dataclass for integrating PDF research ingestion results.
@dataclass
class ResearchConfig:
    """
    Configuration for the PDF research ingestion pipeline.
    Attributes:
        generated_code_dir: Directory where feature code generated from PDFs will be written.
        config_path: Optional JSON configuration file produced by the ingestion pipeline.
    """
    generated_code_dir: str = "features"
    config_path: str = "forex_research/config.json"
