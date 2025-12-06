import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import ModelConfig
from utils.attention_optimization import (
    OptimizedMultiHeadAttention,
    SlidingWindowAttention,
    AdaptiveAttention,
    TemporalAttention,
    create_optimized_attention,
    MultiHeadTemporalAttention
)


class SharedEncoder(nn.Module):
    """Unified CNN + LSTM + Attention encoder base class for all model variants.
    
    Encapsulates the core pattern: temporal local features (CNN) + sequential
    dependencies (LSTM) + context aggregation (attention) → single embedding.
    
    All model encoders (price sequences, signals, regimes) inherit this base,
    ensuring consistent architecture and reducing code duplication.
    
    Parameters
    ----------
    num_features : int
        Number of input features (sequence width).
    hidden_size_lstm : int
        LSTM hidden dimension.
    num_layers_lstm : int
        Number of LSTM layers. Default: 1.
    cnn_num_filters : int
        Number of CNN output filters. Default: 32.
    cnn_kernel_size : int
        CNN kernel size. Default: 3.
    attention_dim : int
        Attention mechanism hidden dimension. Default: 64.
    dropout : float
        Dropout rate. Default: 0.1.
    bidirectional : bool
        Whether LSTM is bidirectional. Default: True.
    use_optimized_attention : bool
        Whether to use optimized attention for long sequences. Default: False.
    use_multihead_attention : bool
        Whether to use multi-head attention. Default: False.
    n_attention_heads : int
        Number of attention heads (if multihead). Default: 4.
    max_seq_length : int
        Max sequence length for optimized attention. Default: 1024.
    use_adaptive_attention : bool
        Whether attention is adaptive. Default: False.
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_size_lstm: int = 64,
        num_layers_lstm: int = 1,
        cnn_num_filters: int = 32,
        cnn_kernel_size: int = 3,
        attention_dim: int = 64,
        dropout: float = 0.1,
        bidirectional: bool = True,
        use_optimized_attention: bool = False,
        use_multihead_attention: bool = False,
        n_attention_heads: int = 4,
        max_seq_length: int = 1024,
        use_adaptive_attention: bool = False,
    ):
        super().__init__()
        
        # CNN for local temporal patterns
        padding = cnn_kernel_size // 2
        self.cnn = nn.Conv1d(
            in_channels=num_features,
            out_channels=cnn_num_filters,
            kernel_size=cnn_kernel_size,
            padding=padding,
        )
        
        # LSTM for sequential dependencies
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size_lstm,
            num_layers=num_layers_lstm,
            batch_first=True,
            bidirectional=bidirectional,
        )
        
        # Compute combined dimension
        lstm_factor = 2 if bidirectional else 1
        lstm_out_dim = lstm_factor * hidden_size_lstm
        self.output_dim = lstm_out_dim + cnn_num_filters
        
        # Attention for context aggregation
        if use_optimized_attention:
            self.attention = create_optimized_attention(
                input_dim=self.output_dim,
                attention_dim=attention_dim,
                n_heads=n_attention_heads,
                max_seq_length=max_seq_length,
                use_adaptive=use_adaptive_attention,
            )
        elif use_multihead_attention:
            self.attention = MultiHeadTemporalAttention(
                input_dim=self.output_dim,
                attention_dim=attention_dim,
                n_heads=n_attention_heads,
            )
        else:
            self.attention = TemporalAttention(self.output_dim, attention_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: CNN + LSTM fusion → Attention → Embedding.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences [B, T, num_features].
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - context: Aggregated embedding [B, output_dim]
            - attn_weights: Attention weights [B, T] or [B, n_heads, T]
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # [B, T, lstm_out_dim]
        
        # CNN forward (permute for Conv1d: [B, F, T])
        cnn_in = x.permute(0, 2, 1)
        cnn_features = F.relu(self.cnn(cnn_in)).permute(0, 2, 1)  # [B, T, cnn_num_filters]
        
        # Concatenate CNN and LSTM outputs
        combined = torch.cat([lstm_out, cnn_features], dim=-1)  # [B, T, output_dim]
        
        # Apply attention
        context, attn_weights = self.attention(combined)
        
        return self.dropout(context), attn_weights


class PriceSequenceEncoder(SharedEncoder):
    """CNN + (bi)LSTM + temporal attention encoder returning a single embedding.
    
    Backward-compatible wrapper around SharedEncoder for price sequence encoding.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(
            num_features=cfg.num_features,
            hidden_size_lstm=cfg.hidden_size_lstm,
            num_layers_lstm=cfg.num_layers_lstm,
            cnn_num_filters=cfg.cnn_num_filters,
            cnn_kernel_size=cfg.cnn_kernel_size,
            attention_dim=cfg.attention_dim,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
            use_optimized_attention=cfg.use_optimized_attention,
            use_multihead_attention=cfg.use_multihead_attention,
            n_attention_heads=cfg.n_attention_heads,
            max_seq_length=cfg.max_seq_length,
            use_adaptive_attention=cfg.use_adaptive_attention,
        )


class DignityModel(nn.Module):
    """
    CNN + BiLSTM + temporal attention Dignity model with multi-head outputs.
    Enhanced with optimized attention for longer sequences.
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
