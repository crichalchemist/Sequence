# Troubleshooting Guide & FAQ

**Last Updated:** 2025-12-06

---

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Data Pipeline](#data-pipeline)
3. [Model Training](#model-training)
4. [RL Training](#rl-training)
5. [Evaluation](#evaluation)
6. [Performance & Optimization](#performance--optimization)
7. [Common Errors](#common-errors)
8. [FAQ](#faq)

---

## Installation & Setup

### Q: How do I set up the project?

**A:**
```bash
# Clone and enter repo
git clone <repo-url>
cd Sequence

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
```

### Q: My virtual environment isn't working

**A:** Try creating a fresh one:
```bash
# Remove old venv
rm -rf .venv

# Create new venv
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Q: How do I check if CUDA is available?

**A:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

If CUDA is not available, fall back to CPU:
```bash
python train/run_training.py --device cpu
```

---

## Data Pipeline

### Q: I get "ModuleNotFoundError: No module named 'data.agents'"

**A:** Ensure you're running from the repository root:
```bash
cd /path/to/Sequence
python data/prepare_dataset.py ...  # ✓ Correct
# NOT: cd data && python prepare_dataset.py ...  # ✗ Wrong
```

### Q: Data validation fails with "Missing columns"

**A:** Check that your CSV has the required columns:
```python
import pandas as pd

df = pd.read_csv("your_data.csv")
print(df.columns)
# Should contain: datetime, open, high, low, close, [volume]
```

If columns are missing or named differently, preprocess before `prepare_dataset.py`:
```python
df = df.rename(columns={
    "Date": "datetime",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
})
```

### Q: "Duplicate timestamps" warning - is this a problem?

**A:** Duplicates are automatically removed by `deduplicate_on_datetime()`, keeping the first occurrence. If you're concerned about data quality:

```python
from data.agents.base_agent import deduplicate_on_datetime

dups = df.duplicated(subset=["datetime"], keep=False)
print(f"Found {dups.sum()} duplicate timestamps")
df_clean = deduplicate_on_datetime(df)
```

### Q: Features are all NaN after preparation

**A:** Check for invalid OHLC relationships:
```python
invalid = (df["high"] < df["low"]) | (df["high"] < df["open"]) | (df["high"] < df["close"])
print(f"Invalid OHLC rows: {invalid.sum()}")
df_clean = df[~invalid]
```

### Q: Sentiment features aren't being added despite `--include-sentiment`

**A:** Ensure GDELT data is available:
```python
from features.agent_sentiment import attach_sentiment_features
import pandas as pd

try:
    df_with_sentiment = attach_sentiment_features(df, pair="gbpusd")
    print("✓ Sentiment added successfully")
except Exception as e:
    print(f"✗ Sentiment failed: {e}")
    print("  → GDELT data may be unavailable or not downloaded")
```

To download GDELT data first:
```bash
python data/download_gdelt.py --start-date 2020-01-01 --end-date 2025-12-31 --out-dir data/gdelt
```

---

## Model Training

### Q: "CUDA out of memory" error

**A:** Reduce memory usage:

**Option 1: Smaller batch size**
```bash
python train/run_training.py --pairs gbpusd --batch-size 32  # Instead of 64 or 128
```

**Option 2: Enable mixed precision**
```bash
python train/run_training.py --pairs gbpusd --use-amp  # float16 tensors
```

**Option 3: Reduce sequence length**
```bash
python data/prepare_dataset.py --pairs gbpusd --t-in 60  # Instead of 120
```

**Option 4: Fall back to CPU**
```bash
python train/run_training.py --pairs gbpusd --device cpu
```

### Q: Training is very slow

**A:** Enable parallel data loading:
```bash
python train/run_training.py \
  --pairs gbpusd \
  --num-workers 4 \
  --pin-memory \
  --prefetch-factor 4 \
  --use-amp
```

Expected speedup: 2-3x on GPU with these settings.

### Q: Model accuracy is stuck at 33% (random for 3 classes)

**A:** Check if the model is actually training:

1. **Verify loss is decreasing:**
   ```python
   # Look at training output for epoch N step M loss X.XXXX
   # If loss stays constant → gradient issue
   ```

2. **Check learning rate:**
   ```bash
   python train/run_training.py --learning-rate 1e-2  # Try larger LR
   ```

3. **Verify data isn't trivial:**
   ```python
   from data.prepare_dataset import process_pair
   loaders = process_pair("gbpusd", ...)
   x, y = next(iter(loaders["train"]))
   print(f"Batch shape: {x.shape}, Label distribution: {y['primary'].unique()}")
   ```

4. **Check risk manager isn't blocking:**
   ```bash
   python train/run_training.py --disable-risk  # Skip risk gating
   ```

### Q: How do I use early stopping?

**A:** Early stopping is built into training:

```python
from utils.training_checkpoint import EarlyStopping

early_stop = EarlyStopping(patience=3)  # Stop after 3 epochs without improvement

for epoch in range(100):
    train_loss = train_epoch(...)
    val_loss = evaluate(...)
    
    if early_stop(val_loss):
        print(f"Stopped at epoch {epoch}")
        break
```

Or use checkpoint manager:
```python
from utils.training_checkpoint import CheckpointManager

manager = CheckpointManager("models", top_n=3)

for epoch in range(100):
    train(...)
    manager.save(model.state_dict(), score=val_loss, epoch=epoch)
    
best_model = manager.load_best()
```

### Q: Can I use mixed precision (AMP)?

**A:** Yes! Enable with:
```bash
python train/run_training.py --use-amp --device cuda
```

This uses float16 tensors where possible, reducing memory by ~50% and often improving speed by 2-3x.

**Important:** Only use on GPU. For CPU, AMP provides no benefit.

---

## RL Training

### Q: A3C training isn't starting workers

**A:** Check multiprocessing settings:

```python
import torch.multiprocessing as mp

# Force spawn method (more compatible than fork)
mp.set_start_method('spawn', force=True)

# Now run A3C
agent.train()
```

Also verify:
```bash
# Check num_workers doesn't exceed CPU count
python -c "import os; print(f'CPU count: {os.cpu_count()}')"

# Reduce if needed
python rl/run_a3c_training.py --num-workers 2
```

### Q: A3C rewards not increasing

**A:** Check reward shaping:

```python
from execution.simulated_retail_env import SimulatedRetailExecutionEnv

env = SimulatedRetailExecutionEnv()
obs, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Info: {info}")
```

If rewards are always 0 or ~uniform:
- Check environment reward calculation
- Verify action effects (buy/sell/hold) are meaningful
- Increase position size or spread sensitivity

### Q: "Queue is full" error

**A:** Reduce log frequency or worker count:
```bash
python rl/run_a3c_training.py --num-workers 2 --log-interval 2000
```

---

## Evaluation

### Q: Evaluation metrics seem wrong

**A:** Verify on synthetic data first:

```python
from eval.agent_eval import classification_metrics
import numpy as np

# Perfect predictions
logits = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
targets = np.array([0, 1, 2])
metrics = classification_metrics(logits, targets)
assert metrics["accuracy"] == 1.0, "Test failed!"
print("✓ Metrics function working correctly")
```

### Q: Per-regime performance differs widely

**A:** This is expected! Some market regimes are harder to predict:

```python
# Check regime distribution
from features.regime_detection import RegimeDetector

detector = RegimeDetector()
df_with_regimes = detector.fit_predict(df)
print(df_with_regimes["regime"].value_counts())

# If one regime has very few samples, accuracy variance is normal
```

---

## Performance & Optimization

### Q: How do I profile training speed?

**A:**
```python
import time
from torch.profiler import profile, record_function

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for batch in train_loader:
        output = model(batch[0])
        loss = compute_loss(output, batch[1])
        loss.backward()

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

### Q: My GPU isn't being used during training

**A:** Check:
```python
import torch

print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# In training:
device = torch.device("cuda:0")
model.to(device)
x = x.to(device)  # Don't forget data!
```

### Q: How do I benchmark different configurations?

**A:**
```bash
# Baseline
time python train/run_training.py --pairs gbpusd --epochs 1 --batch-size 64

# With AMP
time python train/run_training.py --pairs gbpusd --epochs 1 --batch-size 64 --use-amp

# With DataLoader optimization
time python train/run_training.py --pairs gbpusd --epochs 1 --batch-size 64 \
  --num-workers 4 --pin-memory --prefetch-factor 4 --use-amp
```

---

## Common Errors

### `AttributeError: 'NoneType' object has no attribute...`

**Cause:** Model returned None for tensor operations  
**Fix:**
```python
# Check model.forward() returns expected tuple
outputs, attn_weights = model(x)
assert outputs is not None
assert attn_weights is not None
```

### `RuntimeError: Expected all tensors to be on the same device`

**Cause:** Mixing CPU and GPU tensors  
**Fix:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
x = x.to(device)
y = y.to(device)
```

### `FileNotFoundError: [Errno 2] No such file or directory: 'pairs.csv'`

**Cause:** Running from wrong directory  
**Fix:**
```bash
# Always run from repo root
cd /path/to/Sequence
python data/download_all_fx_data.py
```

### `KeyError: 'datetime'` during data loading

**Cause:** CSV doesn't have 'datetime' column  
**Fix:**
```python
df = pd.read_csv("file.csv")
print(df.columns)  # Check column names
df = df.rename(columns={"Date": "datetime"})
```

### `ImportError: cannot import name 'convert_to_utc_and_dedup'`

**Cause:** Old import path still being used  
**Fix:**
```python
# OLD: from utils.datetime import convert_to_utc_and_dedup
# NEW:
from utils.datetime_utils import convert_to_utc_and_dedup
```

---

## FAQ

### Q: What's the minimum data I need to train?

**A:** At least 1,000 bars (1-min OHLCV data = ~17 hours of trading). Recommended: 10,000+ bars.

```python
len_required = t_in + t_out + 100  # 100 for safety
print(f"Minimum bars needed: {len_required}")
```

### Q: How long does training take?

**A:**
- Single pair, 10 epochs, 10K bars: ~2-5 min (GPU), ~10-30 min (CPU)
- Multi-pair (10 pairs), 10 epochs: ~20-50 min (GPU), 2-5 hours (CPU)

### Q: Should I train on GPU or CPU?

**A:** Use GPU if available. GPU is typically 3-10x faster than CPU.

```bash
python train/run_training.py --device cuda  # GPU
python train/run_training.py --device cpu   # CPU (fallback)
```

### Q: Can I resume training from a checkpoint?

**A:** Yes, load the checkpoint manually:
```python
model = build_model(cfg)
checkpoint = torch.load("models/gbpusd_best.pt")
model.load_state_dict(checkpoint["model_state"])
# Continue training with same model
```

### Q: How do I prevent overfitting?

**A:**
1. Use `early_stop_patience` to halt on plateau
2. Increase `dropout` in ModelConfig
3. Use `weight_decay` in TrainingConfig
4. Ensure sufficient validation data (val_ratio ≥ 0.15)

### Q: What if my evaluation metrics don't match my test dataset?

**A:** Ensure consistent preprocessing:
```python
# SAME config for prep and eval
data_cfg = DataConfig(...)
agent_prep = SingleTaskDataAgent(data_cfg)

# Load same splits
datasets_prep = agent_prep.load_and_split(csv_file)
datasets_eval = agent_prep.load_and_split(csv_file)  # Use SAME agent

assert torch.allclose(datasets_prep["test"].sequences, datasets_eval["test"].sequences)
```

### Q: How do I experiment with different architectures?

**A:**
```python
# Change ModelConfig
model_cfg = ModelConfig(
    num_features=20,
    hidden_size_lstm=128,  # Larger
    cnn_num_filters=64,    # More filters
    attention_dim=128,
    use_multihead_attention=True,  # Multi-head
    n_attention_heads=8,
    dropout=0.2,  # More dropout for regularization
)

model = build_model(model_cfg)
history = train_model(model, ...)
```

---

**Need more help?** Check the following resources:

- [Architecture & API Reference](docs/api/ARCHITECTURE_API_REFERENCE.md) - Complete API documentation
- [Backtesting Integration Guide](docs/guides/BACKTESTING_INTEGRATION_GUIDE.md) - RL training with backtesting
- [Tracing Implementation Guide](docs/guides/TRACING_IMPLEMENTATION.md) - Debugging and observability
- [README](README.md) - Project overview and quick start
