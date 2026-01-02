# Sequence Codebase Reorganization & Enhancement Plan

## Executive Summary

This plan addresses critical issues from the recent reorganization (commit 2d1796a) and implements crypto trading support with GPU optimization. The implementation is divided into 5 phases executed sequentially.

## Background Context

### Issues Identified

1. **Broken Imports (32+ files)**: Files import from `config.config` and `features.*` but paths moved to `run/config/config.py` and `train/features/*`
2. **Missing Crypto Integration**: 10 crypto pairs defined but zero prepared datasets, no crypto-specific preprocessing
3. **A3C GPU Bottleneck**: Global model forced to CPU (line 141 in `rl/agents/a3c_agent.py`)
4. **No Multi-GPU Support**: No DataParallel or DistributedDataParallel implementation
5. **Fragmented Training Scripts**: Multiple entry points with unclear integration

### User Requirements

- Fix all imports to use correct paths
- Asset-class aware preprocessing (FX vs Crypto with configurable parameters)
- Jupyter notebook with full pipeline, multi-GPU/AMP, visualization
- Enable GPU-based A3C training
- Revise README for transparency

---

## Phase 1: Fix Import Paths (CRITICAL - Blocks All Other Work)

### Objective
Update 32+ files to use correct import paths after reorganization.

### Files to Modify

#### Group A: Files Missing sys.path Setup (5 files)
Add sys.path initialization to make imports work:

1. `/train/bayesian_tuning.py`
2. `/train/hyperparameter_tuning.py`
3. `/train/training_manager.py`
4. `/train/loss_weighting.py`
5. `/train/core/agent_train.py`

**Change Pattern**:
```python
# Add after existing imports at top of file:
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # or parents[2] for core/ subdirectory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

#### Group B: Files with Config Import (14+ files)
These already have sys.path setup, imports work due to path manipulation:

- `/train/run_training.py` (line 33-39)
- `/train/run_training_multitask.py` (similar)
- `/rl/run_a3c_training.py` (line 14)
- `/eval/run_evaluation.py` (line 29)
- `/eval/ensemble_timesfm.py` (line 32)
- All other files importing `from config.config`

**Decision**: KEEP AS-IS. These work because `run/` is on `sys.path`. Changing to `from run.config.config` is unnecessary and breaks the established pattern.

#### Group C: Files with Features Import (14+ files)
Update to use correct path:

1. `/data/prepare_dataset.py` (lines 28-30)
2. `/data/prepare_multitask_dataset.py` (lines 43-45)
3. `/eval/ensemble_timesfm.py` (line 35)
4. `/run/scripts/validate_training_data.py` (lines 23-25)
5. `/tests/test_agent_features.py` (line 12)
6. Plus 9 more files

**Change Pattern**:
```python
# OLD:
from features.agent_features import build_feature_frame
from features.agent_sentiment import aggregate_sentiment, attach_sentiment_features
from features.intrinsic_time import build_intrinsic_time_bars

# NEW:
from train.features.agent_features import build_feature_frame
from train.features.agent_sentiment import aggregate_sentiment, attach_sentiment_features
from train.features.intrinsic_time import build_intrinsic_time_bars
```

### Testing Strategy
After Phase 1, run:
```bash
python -m pytest tests/test_agent_features.py
python data/prepare_dataset.py --pairs gbpusd --t-in 60 --t-out 10 --task-type classification
python train/run_training.py --pairs gbpusd --epochs 1
```

---

## Phase 2: Asset-Class Aware Preprocessing

### Objective
Create flexible preprocessing system that detects asset class and applies appropriate parameters.

### 2.1: Add AssetClass Detection

**File**: `/run/config/config.py`

Add new configuration class:
```python
from enum import Enum

class AssetClass(str, Enum):
    FX = "fx"
    CRYPTO = "crypto"
    COMMODITY = "commodity"

@dataclass
class AssetConfig:
    """Asset-class specific configuration."""
    asset_class: AssetClass = AssetClass.FX

    # Volatility windows (crypto uses shorter due to 10x volatility)
    sma_windows_fx: list[int] = field(default_factory=lambda: [10, 20, 50])
    sma_windows_crypto: list[int] = field(default_factory=lambda: [5, 10, 20])

    ema_windows_fx: list[int] = field(default_factory=lambda: [10, 20, 50])
    ema_windows_crypto: list[int] = field(default_factory=lambda: [5, 10, 20])

    # RSI windows
    rsi_window_fx: int = 14
    rsi_window_crypto: int = 7  # Faster for crypto

    # Bollinger bands
    bollinger_window_fx: int = 20
    bollinger_window_crypto: int = 10

    # ATR windows
    atr_window_fx: int = 14
    atr_window_crypto: int = 7

    # Directional change thresholds
    dc_threshold_fx: float = 0.0005  # 5 pips for FX
    dc_threshold_crypto: float = 0.005  # 0.5% for crypto (10x more volatile)

    # Market hours (for timezone handling)
    market_24_7: bool = False  # True for crypto

    def get_feature_config(self) -> FeatureConfig:
        """Get FeatureConfig appropriate for asset class."""
        if self.asset_class == AssetClass.CRYPTO:
            return FeatureConfig(
                sma_windows=self.sma_windows_crypto,
                ema_windows=self.ema_windows_crypto,
                rsi_window=self.rsi_window_crypto,
                bollinger_window=self.bollinger_window_crypto,
                atr_window=self.atr_window_crypto,
                dc_threshold_up=self.dc_threshold_crypto,
            )
        else:  # FX
            return FeatureConfig(
                sma_windows=self.sma_windows_fx,
                ema_windows=self.ema_windows_fx,
                rsi_window=self.rsi_window_fx,
                bollinger_window=self.bollinger_window_fx,
                atr_window=self.atr_window_fx,
                dc_threshold_up=self.dc_threshold_fx,
            )

    @staticmethod
    def detect_from_pair(pair: str) -> AssetClass:
        """Detect asset class from pair name."""
        pair_upper = pair.upper()

        # Crypto pairs contain common crypto symbols
        crypto_bases = {'BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'DOGE', 'LTC', 'LINK', 'MATIC', 'AVAX', 'BNB', 'DOT'}
        if any(base in pair_upper for base in crypto_bases):
            return AssetClass.CRYPTO

        # Commodities
        if 'XAU' in pair_upper or 'XAG' in pair_upper or 'GC=' in pair_upper:
            return AssetClass.COMMODITY

        return AssetClass.FX
```

### 2.2: Update prepare_dataset.py

**File**: `/data/prepare_dataset.py`

Add asset detection and config selection (insert after line 73):
```python
def prepare_features_for_pair(pair: str, df: pd.DataFrame, args) -> pd.DataFrame:
    """Prepare features with asset-class aware config."""
    # Detect asset class
    asset_class = AssetConfig.detect_from_pair(pair)
    asset_cfg = AssetConfig(asset_class=asset_class)

    # Get appropriate feature config
    feature_cfg = asset_cfg.get_feature_config()

    # Override with CLI args if provided
    if hasattr(args, 'sma_windows') and args.sma_windows:
        feature_cfg.sma_windows = [int(x) for x in args.sma_windows.split(',')]
    # ... repeat for other args

    logger.info(f"Pair {pair} detected as {asset_class.value}, using config: {feature_cfg}")

    # Build features with asset-specific config
    df_features = build_feature_frame(df, feature_cfg)

    return df_features, asset_class
```

### 2.3: Integrate Crypto Data Sources

**File**: `/data/prepare_dataset.py`

Modify `_load_pair_data()` to handle both HistData zips and crypto CSVs:
```python
def _load_pair_data(pair: str, input_root: Path, years: list[str] | None, asset_class: AssetClass) -> pd.DataFrame:
    """Load pair data from appropriate source."""

    # For crypto, check crypto_test directory first
    if asset_class == AssetClass.CRYPTO:
        crypto_test_path = ROOT / "data" / "crypto_test" / pair.lower()
        if crypto_test_path.exists():
            logger.info(f"Loading crypto data from {crypto_test_path}")
            csv_files = list(crypto_test_path.glob("*.csv"))
            if csv_files:
                dfs = []
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file, sep=';', header=None,
                                   names=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S')
                    dfs.append(df)
                combined = pd.concat(dfs, ignore_index=True)
                return validate_dataframe(combined, ['datetime', 'open', 'high', 'low', 'close', 'volume'])

    # Original HistData zip loading for FX...
    # (keep existing implementation)
```

### Testing Strategy
```bash
# Test FX preprocessing (should use windows: 10, 20, 50)
python data/prepare_dataset.py --pairs gbpusd --t-in 60 --t-out 10

# Test crypto preprocessing (should use windows: 5, 10, 20)
python data/prepare_dataset.py --pairs btcusd --t-in 60 --t-out 10

# Verify different configs applied
grep "using config" logs/preprocessing.log
```

---

## Phase 3: Enable GPU-Based A3C Training

### Objective
Fix A3C to use GPU for global model instead of forcing CPU.

### 3.1: Modify A3C Agent Architecture

**File**: `/rl/agents/a3c_agent.py`

**Current Issue** (lines 138-143):
```python
self.device = torch.device(...)
self.shared_device = torch.device("cpu")  # ❌ FORCED CPU
self.global_model = ActorCriticNetwork(...).to(self.shared_device)  # On CPU
self.global_model.share_memory()  # Share CPU memory
```

**Proposed Solution**:

Replace lines 138-151 with:
```python
self.device = torch.device(
    a3c_cfg.device if a3c_cfg.device != "cuda" or torch.cuda.is_available() else "cpu"
)

# Use GPU for global model if available
self.shared_device = self.device  # ✓ Use same device as workers

# Create global model on shared device
self.global_model = ActorCriticNetwork(model_cfg, action_dim).to(self.shared_device)

# Share memory ONLY if on CPU (required for multiprocessing)
if self.shared_device.type == "cpu":
    self.global_model.share_memory()
    logger.info("A3C using CPU with shared memory")
else:
    logger.info(f"A3C using GPU: {self.shared_device}")
    # For GPU, workers will sync via explicit .to() transfers
```

### 3.2: Modify Worker Process

**File**: `/rl/agents/a3c_agent.py`

Update `worker_process()` method (around line 240-260) to handle GPU gradient sync:
```python
def _sync_gradients(self, local_model, global_model):
    """Sync gradients from local to global model."""
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        if global_param.grad is not None:
            return  # Already accumulated

        # Transfer gradient to global device
        if local_param.grad is not None:
            grad = local_param.grad.to(self.shared_device)  # Move to global device
            global_param._grad = grad  # Assign gradient
```

**Change in worker_process** (around line 250):
```python
# OLD:
for local_p, global_p in zip(local_model.parameters(), self.global_model.parameters()):
    global_p._grad = local_p.grad.cpu()  # ❌ Forces CPU transfer

# NEW:
self._sync_gradients(local_model, self.global_model)  # ✓ Device-aware
```

### 3.3: Update Config Default

**File**: `/rl/agents/a3c_agent.py`

Change A3CConfig default (line 40):
```python
device: str = "cuda"  # Changed from "cpu"
```

### Testing Strategy
```bash
# Test A3C on GPU
python rl/run_a3c_training.py \
  --pair btcusd \
  --env-mode simulated \
  --num-workers 4 \
  --total-steps 10000 \
  --device cuda

# Monitor GPU usage
nvidia-smi -l 1
```

---

## Phase 4: Add Multi-GPU & AMP Support

### Objective
Enable DataParallel for multi-GPU training and integrate existing AMP utilities.

### 4.1: Create Multi-GPU Wrapper

**File**: `/utils/multi_gpu.py` (NEW FILE)

```python
"""Multi-GPU training utilities."""

import torch
import torch.nn as nn
from typing import Optional

def setup_multi_gpu(
    model: nn.Module,
    device: str = "cuda",
    device_ids: Optional[list[int]] = None
) -> tuple[nn.Module, str]:
    """Setup model for multi-GPU training.

    Args:
        model: PyTorch model
        device: Target device ("cuda" or "cpu")
        device_ids: List of GPU IDs to use (None = all available)

    Returns:
        (wrapped_model, device_str)
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()

        if gpu_count > 1:
            if device_ids is None:
                device_ids = list(range(gpu_count))

            print(f"Using DataParallel on {len(device_ids)} GPUs: {device_ids}")
            model = nn.DataParallel(model, device_ids=device_ids)
            device_str = f"cuda:{device_ids[0]}"  # Primary device
        else:
            print("Single GPU detected")
            device_str = "cuda:0"

        model = model.to(device_str)
    else:
        print("Using CPU")
        device_str = "cpu"
        model = model.to(device_str)

    return model, device_str


def get_unwrapped_model(model: nn.Module) -> nn.Module:
    """Get underlying model from DataParallel wrapper."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model
```

### 4.2: Integrate into Training Scripts

**File**: `/train/core/agent_train.py`

Modify `train_model()` function (around line 50-60):
```python
from utils.amp import AMPTrainer
from utils.multi_gpu import setup_multi_gpu, get_unwrapped_model

def train_model(...):
    # Setup multi-GPU
    model, device_str = setup_multi_gpu(model, device=cfg.device)

    # Setup AMP
    amp_trainer = AMPTrainer(
        enabled=cfg.use_amp if hasattr(cfg, 'use_amp') else False,
        device=device_str
    )

    # Training loop
    for epoch in range(cfg.epochs):
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device_str), y.to(device_str)

            # Forward pass with AMP
            with amp_trainer.autocast_context():
                logits = model(x)
                loss = criterion(logits, y)

            # Backward with gradient scaling
            optimizer.zero_grad()
            amp_trainer.backward(loss, optimizer, max_grad_norm=cfg.max_grad_norm)

    # Save unwrapped model
    torch.save({
        'model_state': get_unwrapped_model(model).state_dict(),
        ...
    }, checkpoint_path)
```

### 4.3: Update Config

**File**: `/run/config/config.py`

Add to `TrainingConfig`:
```python
@dataclass
class TrainingConfig:
    # ... existing fields ...
    use_amp: bool = False  # Enable automatic mixed precision
    fp16: bool = False  # Use FP16 instead of FP32
    device_ids: list[int] | None = None  # GPU IDs (None = all)
```

### Testing Strategy
```bash
# Test multi-GPU training
python train/run_training.py \
  --pairs gbpusd \
  --epochs 5 \
  --batch-size 128 \
  --use-amp \
  --device cuda

# Monitor GPU utilization across devices
nvidia-smi dmon -s u
```

---

## Phase 5: Create Jupyter Notebook

### Objective
Create comprehensive notebook for end-to-end pipeline with visualization.

### 5.1: Notebook Structure

**File**: `/notebooks/training_pipeline_demo.ipynb` (NEW FILE)

**Cell Structure**:

1. **Setup & Imports**
   - Install dependencies
   - Import libraries
   - Check GPU availability

2. **Section 1: Data Download**
   - Download HistData for FX pairs
   - Download Yahoo Finance for crypto
   - Display download statistics

3. **Section 2: Preprocessing**
   - Asset class detection demo
   - Apply FX vs Crypto configs
   - Visualize feature distributions

4. **Section 3: Dataset Preparation**
   - Create train/val/test splits
   - Show batch samples
   - Feature correlation heatmaps

5. **Section 4: Model Training**
   - Setup multi-GPU
   - Enable AMP
   - Train with progress bars (tqdm)
   - Real-time loss/accuracy plots

6. **Section 5: Training Visualization**
   - Training curves (loss, accuracy, F1)
   - GPU utilization timeline
   - Attention weight heatmaps
   - Feature importance

7. **Section 6: Evaluation**
   - Test set metrics
   - Confusion matrix
   - ROC curves
   - Example predictions

8. **Section 7: Crypto-Specific Analysis**
   - Compare FX vs Crypto preprocessing
   - Volatility analysis
   - 24/7 market patterns

### 5.2: Key Notebook Code Snippets

```python
# Cell: GPU Setup
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Cell: Data Download with Progress
from tqdm.notebook import tqdm
pairs = ['gbpusd', 'eurusd', 'btcusd', 'ethusd']
for pair in tqdm(pairs, desc="Downloading pairs"):
    # Download logic...

# Cell: Training with Visualization
from IPython.display import clear_output
import matplotlib.pyplot as plt

losses = []
for epoch in range(epochs):
    epoch_loss = train_epoch(...)
    losses.append(epoch_loss)

    # Real-time plot
    clear_output(wait=True)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.show()

# Cell: GPU Monitoring
import GPUtil
gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU {gpu.id}: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.load*100:.1f}%)")
```

### Testing Strategy
```bash
# Test notebook execution
jupyter nbconvert --to notebook --execute notebooks/training_pipeline_demo.ipynb

# Or run interactively
jupyter notebook notebooks/training_pipeline_demo.ipynb
```

---

## Phase 6: README Revision

### Objective
Audit README against actual codebase and document new features.

### 6.1: Sections to Update

**File**: `/README.md`

1. **Architecture Section** (lines 136-244)
   - Add crypto support documentation
   - Document asset-class aware preprocessing
   - Update data flow diagram to include crypto paths

2. **Quick Start Section** (lines 36-133)
   - Add crypto pair examples
   - Document multi-GPU training
   - Add AMP training examples

3. **Advanced Usage** (lines 246-294)
   - Add asset-class configuration examples
   - Document GPU optimization flags

### 6.2: New Sections to Add

```markdown
## Crypto Trading Support

Sequence now supports cryptocurrency pairs with specialized preprocessing:

### Asset-Class Detection

The framework automatically detects asset class from pair names:
- **FX pairs**: `gbpusd`, `eurusd`, etc.
- **Crypto pairs**: `btcusd`, `ethusd`, `solusd`, etc.

### Crypto-Specific Features

- **Adjusted volatility windows**: 5/10/20 instead of 10/20/50 (accounts for 10x crypto volatility)
- **Faster indicators**: RSI(7), ATR(7), Bollinger(10) vs FX defaults
- **24/7 market handling**: No timezone restrictions
- **Higher DC thresholds**: 0.5% vs 5 pips for FX

### Example Usage

\`\`\`bash
# Train on crypto pair
python train/run_training.py \
  --pairs btcusd,ethusd \
  --epochs 50 \
  --batch-size 64

# Mixed FX and crypto
python train/run_training.py \
  --pairs gbpusd,btcusd \
  --epochs 50
\`\`\`

## Multi-GPU Training

### Enable Multi-GPU

\`\`\`bash
# Use all available GPUs
python train/run_training.py \
  --pairs gbpusd \
  --epochs 50 \
  --device cuda

# Specify GPU IDs
python train/run_training.py \
  --pairs gbpusd \
  --epochs 50 \
  --device cuda \
  --device-ids 0,1,2
\`\`\`

### Enable Mixed Precision (AMP)

\`\`\`bash
# Faster training with reduced memory
python train/run_training.py \
  --pairs gbpusd \
  --epochs 50 \
  --use-amp \
  --batch-size 256  # Can use larger batches
\`\`\`

### GPU-Accelerated A3C

\`\`\`bash
# A3C now uses GPU by default
python rl/run_a3c_training.py \
  --pair btcusd \
  --env-mode backtesting \
  --num-workers 8 \
  --device cuda  # GPU-based global model
\`\`\`
```

### 6.3: Accuracy Checks

Remove or update outdated sections:
- ❌ Remove claim about "automatic multi-GPU support" (was false, now true)
- ✓ Update architecture diagram to show crypto data sources
- ✓ Document actual training entry points vs experimental scripts
- ✓ Clarify difference between `run/training_pipeline.py` and `train/run_training.py`

### Testing Strategy
After updates, validate:
```bash
# Test all README examples
bash -c "$(cat README.md | grep '```bash' -A 5 | grep -v '```')"

# Check links
python -m markdown_link_check README.md
```

---

## Implementation Order & Dependencies

### Execution Sequence

```
Phase 1 (Import Fixes)
  ↓ [BLOCKS ALL OTHER PHASES]
Phase 2 (Asset-Class Preprocessing)
  ↓ [REQUIRED FOR PHASE 5 NOTEBOOK]
Phase 3 (A3C GPU)  [PARALLEL]  Phase 4 (Multi-GPU/AMP)
  ↓                                ↓
  └─────────→ Phase 5 (Jupyter Notebook) ←─────┘
                       ↓
              Phase 6 (README Revision)
```

### Time Estimates

- **Phase 1**: 2-3 hours (mechanical find-replace with testing)
- **Phase 2**: 4-5 hours (new config system + integration)
- **Phase 3**: 2-3 hours (A3C architecture changes)
- **Phase 4**: 3-4 hours (DataParallel wrapper + integration)
- **Phase 5**: 5-6 hours (comprehensive notebook)
- **Phase 6**: 2-3 hours (README audit and updates)

**Total**: 18-24 hours of focused implementation

---

## Risk Mitigation

### High-Risk Changes

1. **A3C Architecture Change (Phase 3)**
   - Risk: May break existing A3C checkpoints
   - Mitigation: Add checkpoint version detection, backward compatibility loader

2. **Import Path Changes (Phase 1)**
   - Risk: Breaks external scripts that import from this repo
   - Mitigation: Test all entry points, document changes in CHANGELOG

3. **Multi-GPU State Dict (Phase 4)**
   - Risk: DataParallel adds `module.` prefix to state dict keys
   - Mitigation: Use `get_unwrapped_model()` helper before saving

### Testing Checkpoints

After each phase:
```bash
# Phase 1
pytest tests/ -v
python -m ruff check .

# Phase 2
python data/prepare_dataset.py --pairs gbpusd,btcusd --t-in 60 --t-out 10
ls data/prepared/  # Verify both FX and crypto datasets

# Phase 3
python rl/run_a3c_training.py --pair gbpusd --total-steps 1000 --device cuda
nvidia-smi  # Verify GPU usage

# Phase 4
python train/run_training.py --pairs gbpusd --epochs 2 --use-amp --device cuda
# Check multi-GPU batch splitting

# Phase 5
jupyter nbconvert --to notebook --execute notebooks/training_pipeline_demo.ipynb

# Phase 6
# Manually validate all README examples
```

---

## Critical Files Summary

### Will Be Modified

| File | Phase | Changes |
|------|-------|---------|
| `/run/config/config.py` | 2 | Add AssetConfig, AssetClass enum |
| `/data/prepare_dataset.py` | 1, 2 | Fix imports, add asset detection |
| `/data/prepare_multitask_dataset.py` | 1, 2 | Fix imports, add asset detection |
| `/rl/agents/a3c_agent.py` | 1, 3 | Fix imports, enable GPU |
| `/train/core/agent_train.py` | 1, 4 | Add sys.path, multi-GPU/AMP |
| `/train/bayesian_tuning.py` | 1 | Add sys.path setup |
| `/train/hyperparameter_tuning.py` | 1 | Add sys.path setup |
| `/train/training_manager.py` | 1 | Add sys.path setup |
| `/train/loss_weighting.py` | 1 | Add sys.path setup |
| `/eval/ensemble_timesfm.py` | 1 | Fix features imports |
| `/run/scripts/validate_training_data.py` | 1 | Fix features imports |
| `/tests/test_agent_features.py` | 1 | Fix features imports |
| `/README.md` | 6 | Add crypto, multi-GPU docs |

### Will Be Created

| File | Phase | Purpose |
|------|-------|---------|
| `/utils/multi_gpu.py` | 4 | Multi-GPU wrapper utilities |
| `/notebooks/training_pipeline_demo.ipynb` | 5 | End-to-end training demo |

---

## Success Criteria

1. ✅ All 32+ import errors resolved
2. ✅ Crypto pairs (btcusd, ethusd) successfully preprocessed
3. ✅ A3C training uses GPU (verify via `nvidia-smi`)
4. ✅ Multi-GPU training splits batches across devices
5. ✅ AMP training reduces memory usage (verify larger batch sizes work)
6. ✅ Jupyter notebook runs end-to-end without errors
7. ✅ README examples all execute successfully
8. ✅ All existing tests pass

---

## Post-Implementation Validation

```bash
# Full integration test
./scripts/run_full_integration_test.sh

# Contents of test script:
python data/prepare_dataset.py --pairs gbpusd,btcusd --t-in 120 --t-out 10
python train/run_training.py --pairs gbpusd --epochs 2 --use-amp --device cuda
python rl/run_a3c_training.py --pair btcusd --total-steps 5000 --device cuda
python eval/run_evaluation.py --pairs gbpusd --checkpoint-path models/gbpusd_best_model.pt
jupyter nbconvert --to notebook --execute notebooks/training_pipeline_demo.ipynb
pytest tests/ -v
```
