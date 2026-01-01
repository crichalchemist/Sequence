# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Sequence** is a deep learning framework for FX market prediction combining CNN-LSTM-Attention hybrid models, intrinsic time representations, sentiment analysis (GDELT + FinBERT), and reinforcement learning (A3C) for algorithmic trading. The codebase supports end-to-end workflows from raw data ingestion to trained trading policies.

## Common Commands

### Data Preparation

```bash
# Prepare dataset with intrinsic time transformation
python data/prepare_dataset.py \
  --pairs gbpusd \
  --t-in 120 \
  --t-out 10 \
  --task-type classification \
  --intrinsic-time \
  --dc-threshold-up 0.0005

# Enable GDELT sentiment enrichment
python data/prepare_dataset.py \
  --pairs gbpusd \
  --t-in 120 \
  --t-out 10 \
  --include-sentiment
```

### Training

```bash
# Supervised learning (classification)
python train/run_training.py \
  --pairs gbpusd \
  --epochs 50 \
  --learning-rate 1e-3 \
  --batch-size 64

# Multi-task learning (price + volatility + regime)
python train/run_training_multitask.py \
  --pairs gbpusd \
  --epochs 50 \
  --batch-size 64

# Reinforcement learning (A3C with backtesting)
python rl/run_a3c_training.py \
  --pair gbpusd \
  --env-mode backtesting \
  --historical-data data/data/gbpusd/gbpusd_prepared.csv \
  --num-workers 8 \
  --total-steps 1000000

# Unified pipeline (download → prepare → train → RL)
python run/training_pipeline.py \
  --pairs gbpusd \
  --run-histdata-download \
  --epochs 50 \
  --run-rl-training \
  --rl-env-mode backtesting \
  --rl-num-workers 8
```

### Evaluation

```bash
# Evaluate trained model
python eval/run_evaluation.py \
  --pairs gbpusd \
  --checkpoint-path models/gbpusd_best_model.pt

# Ensemble with TimesFM foundation model
python eval/ensemble_timesfm.py \
  --pairs gbpusd \
  --years 2023 \
  --t-in 120 \
  --t-out 10 \
  --checkpoint-root models \
  --device cuda
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_integration_full_pipeline.py

# Run with verbose output
pytest -v tests/

# Run tests matching pattern
pytest -k "test_data" tests/
```

### Code Quality

```bash
# Lint codebase
ruff check .

# Format code
ruff format .

# Lint specific files
ruff check train/ models/

# Fix auto-fixable issues
ruff check --fix .
```

## Architecture Overview

### Core Components

The codebase follows a modular architecture with clear separation of concerns:

1. **Data Pipeline** (`data/`, `train/features/`)
   - `prepare_dataset.py`: Main data preparation entry point
   - `intrinsic_time.py`: Directional-change time transformation
   - `iterable_dataset.py`: Memory-efficient PyTorch dataset
   - `gdelt/`: GDELT news event download and parsing
   - `train/features/agent_features.py`: Technical indicator computation
   - `train/features/agent_sentiment.py`: FinBERT sentiment integration

2. **Model Architecture** (`models/`)
   - `agent_hybrid.py`: SharedEncoder base class (CNN→LSTM→Attention)
   - `agent_multitask.py`: Multi-task variant (price + volatility + regime)
   - `signal_policy.py`: Signal model and execution policy components
   - `regime_encoder.py`: Market regime classifier

3. **Training System** (`train/`)
   - `run_training.py`: Main supervised learning entry point
   - `run_training_multitask.py`: Multi-task training
   - `core/agent_train.py`: Core training logic (pretrain_signal_model, train_execution_policy)
   - `training_manager.py`: Training orchestration with checkpointing
   - `loss_weighting.py`: Dynamic loss weighting for multi-task learning
   - `hyperparameter_tuning.py`: Bayesian hyperparameter optimization

4. **Reinforcement Learning** (`rl/`)
   - `run_a3c_training.py`: A3C training entry point
   - `agents/a3c_agent.py`: Asynchronous Advantage Actor-Critic implementation
   - `agents/sac_agent.py`: Soft Actor-Critic implementation
   - `replay_buffer.py`: Experience replay for off-policy RL

5. **Execution Environments** (`train/execution/`)
   - `simulated_retail_env.py`: Stochastic execution simulator with spread/slippage
   - `backtesting_env.py`: Deterministic historical replay environment
   - `backtest_manager.py`: Backtesting coordination and metrics

6. **Evaluation** (`eval/`)
   - `run_evaluation.py`: Model evaluation entry point
   - `agent_eval.py`: Evaluation metrics (accuracy, precision, recall, F1)
   - `ensemble_timesfm.py`: Ensemble with Google TimesFM foundation model

7. **Utilities** (`utils/`)
   - `attention_optimization.py`: Optimized attention mechanisms
   - `async_checkpoint.py`: Non-blocking checkpoint saving
   - `tracing.py`: OpenTelemetry observability integration
   - `cache_manager.py`: Feature caching for preprocessing
   - `memory_profiler.py`: Memory usage tracking

8. **Configuration** (`run/config/`)
   - `config.py`: Dataclass configurations (DataConfig, ModelConfig, TrainingConfig, etc.)
   - `arg_parser.py`: CLI argument parsing utilities
   - `constants.py`: Global constants

### Key Architectural Patterns

**SharedEncoder Pattern**: All model variants (supervised, multi-task, RL) inherit from `SharedEncoder` in `models/agent_hybrid.py`, providing:
- Temporal local features via 1D CNN layers
- Sequential dependencies via bidirectional LSTM
- Context aggregation via multi-head attention
- Unified embedding for downstream tasks

**Configuration System**: The `run/config/config.py` module defines dataclass configs imported throughout:
- `DataConfig`: Dataset parameters (t_in, t_out, train/val/test splits)
- `FeatureConfig`: Feature engineering toggles (SMA/EMA windows, RSI, Bollinger bands)
- `ModelConfig`: Neural architecture hyperparameters (LSTM hidden size, CNN filters, attention dim)
- `TrainingConfig`: Training parameters (epochs, batch size, learning rate, optimizer)
- `RiskConfig`: Risk management constraints (position limits, drawdown thresholds)
- `ExecutionConfig`: Trading environment settings (spreads, slippage, latency)

**Intrinsic Time Transformation**: The `train/features/intrinsic_time.py` module implements directional-change (DC) bars, replacing fixed-time sampling with event-driven bars triggered by price movements exceeding thresholds. This provides scale-invariant market structure representation.

**Two-Stage Training**:
1. **Signal Model Pretraining** (`pretrain_signal_model` in `train/core/agent_train.py`): Train CNN-LSTM-Attention on supervised price prediction
2. **Execution Policy Training** (`train_execution_policy`): Train RL policy on top of frozen signal model embeddings for optimal execution

**Environment Modes**:
- **Simulated**: Stochastic retail execution with configurable spreads, slippage, latency (`train/execution/simulated_retail_env.py`)
- **Backtesting**: Deterministic historical replay using `backtesting.py` library (`train/execution/backtesting_env.py`)

## Data Flow

```
HistData CSVs (Central time)
  ↓ data/prepare_dataset.py
UTC conversion + deduplication
  ↓ train/features/agent_features.py
Technical indicators (SMA, EMA, RSI, ATR, Bollinger, etc.)
  ↓ train/features/intrinsic_time.py (optional)
Directional-change bars (if --intrinsic-time)
  ↓ train/features/agent_sentiment.py (optional)
GDELT sentiment features (if --include-sentiment)
  ↓ data/iterable_dataset.py
PyTorch IterableDataset (train/val/test splits)
  ↓ train/core/agent_train.py
Model training (supervised or RL)
  ↓ eval/agent_eval.py
Evaluation metrics
```

## Important Development Notes

### Configuration Import Pattern

Always import configs from `run/config/config.py`, NOT from `config/config.py` (which doesn't exist):

```python
from config.config import ModelConfig, TrainingConfig  # WRONG
from run.config.config import ModelConfig, TrainingConfig  # CORRECT
```

However, the import alias `config.config` is used throughout the codebase via the fact that `run/` is on `sys.path`:

```python
from config.config import ModelConfig  # Works because run/ is on sys.path
```

### Feature Engineering Extensibility

The `train/features/agent_features.py` module supports dynamic feature loading from `train/features/generated/` for research-generated features. Add custom features by:
1. Creating a `.py` file in `train/features/generated/`
2. Defining a function with signature `fn(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame`
3. Features are auto-loaded via `_load_generated_features()`

### Intrinsic Time Thresholds

Directional-change thresholds are FX pair-specific. Typical values:
- Major pairs (EUR/USD, GBP/USD): 0.0005–0.001 (5–10 pips)
- Volatile pairs (GBP/JPY): 0.001–0.002 (10–20 pips)
- Crypto pairs: 0.005–0.01 (0.5%–1%)

### GDELT Sentiment Pipeline

Multi-stage process in `data/gdelt/`:
1. `consolidated_downloader.py`: Download GDELT GKG files
2. `parser.py`: Parse news events and filter for FX relevance
3. `train/features/agent_sentiment.py`: Run FinBERT sentiment analysis
4. `alignment.py`: Temporal alignment with OHLCV data

### RL Environment Selection

- Use **backtesting mode** for reproducible experiments and paper results
- Use **simulated mode** for testing robustness to execution noise
- Historical data path format: `data/data/{pair}/{pair}_prepared.csv`

### Multi-GPU Training

Training scripts automatically detect and use all available GPUs via `torch.nn.DataParallel`. Batch size is split across GPUs. No code changes required.

### Checkpoint Management

Checkpoints are saved to `models/` by default:
- `best_model.pt`: Best validation loss checkpoint
- `checkpoint_epoch_{N}/`: Full training state for resumption
- Use `--resume-from-checkpoint` to continue training

### Tracing and Observability

OpenTelemetry tracing is available in `utils/tracing.py`. Initialize with:

```python
from utils.tracing import setup_tracing
setup_tracing(
    service_name="sequence-training",
    otlp_endpoint="http://localhost:4318",
    environment="development"
)
```

## Testing Strategy

Tests follow a multi-level approach:
- **Unit tests**: `test_agent_features.py`, `test_gdelt_parser.py`
- **Integration tests**: `test_integration_full_pipeline.py`, `test_phase1_phase2_integration.py`
- **End-to-end tests**: `test_end_to_end_phases_1_2_3.py`

Key test files:
- `test_iterable_dataset.py`: Dataset loading and batching
- `test_data_splits.py`: Train/val/test split validation
- `test_optimized_attention_integration.py`: Attention mechanism correctness
- `test_gdelt_alignment.py`: GDELT-OHLCV temporal alignment

## Dependencies

Core dependencies (see `requirements.txt`):
- `torch>=2.0.0`: PyTorch deep learning framework
- `pandas>=1.5.0`: Data manipulation
- `numpy>=1.24,<2.0`: Numerical computing
- `transformers>=4.38.0`: FinBERT sentiment model
- `backtesting>=0.3.2`: Backtesting library integration
- `models/timesFM`: Google TimesFM foundation model (editable install via `-e ./models/timesFM`)

Development tools:
- `pytest>=7.4.0`: Testing framework
- `ruff>=0.1.0`: Linting and formatting

## File Organization Conventions

- **Scripts**: Top-level runnable scripts are in `train/`, `rl/`, `eval/`, `data/`
- **Core logic**: Reusable components in module subdirectories (`train/core/`, `train/features/`)
- **Tests**: Mirror source structure in `tests/` (e.g., `train/core/agent_train.py` → `tests/test_phase1_phase2_integration.py`)
- **Configs**: All configuration dataclasses in `run/config/config.py`
- **Utils**: Cross-cutting utilities in `utils/`

## Common Pitfalls

1. **Import Paths**: Always add `ROOT` to `sys.path` in scripts:
   ```python
   ROOT = Path(__file__).resolve().parents[1]
   if str(ROOT) not in sys.path:
       sys.path.insert(0, str(ROOT))
   ```

2. **Time Zones**: HistData CSVs are in US Central time. Use `utils.datetime_utils.convert_to_utc_and_dedup()` for UTC conversion.

3. **Feature Column Order**: Feature columns must match between training and inference. Use `DataConfig.feature_columns` to enforce consistency.

4. **GDELT Mirror**: Default GDELT endpoint can be slow. Use `--gdelt-mirror` for custom mirrors.

5. **Memory Usage**: For large datasets, use `data/iterable_dataset.py` (streaming) instead of loading full data into memory.

6. **Attention Sequence Length**: For sequences >1024, enable `use_optimized_attention=True` in ModelConfig to avoid OOM errors.