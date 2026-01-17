# Google Colab Data Collection Notebook

This directory contains notebooks for working with the Sequence framework on Google Colab.

## 📓 Notebooks

### `colab_data_collection.ipynb` (NEW)
**Purpose:** Complete data collection and training pipeline using Google Drive for storage

**Use when:**
- You need to collect large amounts of FX data
- Local storage is limited
- You want to leverage free Colab GPUs for training
- You need persistent storage across Colab sessions

**Features:**
- ✅ Mount Google Drive for unlimited* storage
- ✅ Collect HistData, GDELT, and fundamental economic data
- ✅ Prepare datasets with feature engineering
- ✅ Train models on Colab GPUs (T4/P100/V100)
- ✅ Store everything on Drive (no local space limits)
- ✅ Resume work across sessions

### `training_guide.ipynb` (Existing)
**Purpose:** General training guide for local development

**Use when:**
- Working in local environment or codespace
- Data is already prepared
- You have sufficient local storage

---

## 🚀 Quick Start: Colab Data Collection

### 1. Open in Google Colab

**Option A:** Direct upload
1. Upload `colab_data_collection.ipynb` to Google Drive
2. Right-click → Open with → Google Colaboratory

**Option B:** From GitHub
1. Open [Google Colab](https://colab.research.google.com/)
2. File → Open notebook → GitHub
3. Enter: `crichalchemist/Sequence`
4. Select: `notebooks/colab_data_collection.ipynb`

### 2. Set Up Google Drive

Run the first cell to mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

This creates the following structure on your Drive:
```
MyDrive/Sequence/
├── data/
│   ├── raw/
│   │   ├── histdata/       # Tick-by-tick price data
│   │   ├── gdelt/          # News sentiment data
│   │   └── fundamentals/   # Economic indicators
│   └── prepared/           # Feature-engineered datasets
├── models/                 # Trained model checkpoints
├── logs/                   # Training logs
└── config/                 # API keys (secure storage)
```

### 3. Configure API Keys (Required for Fundamentals)

**Get FRED API Key (Free):**
1. Visit: https://fred.stlouisfed.org/docs/api/api_key.html
2. Sign up and request API key
3. Store in Colab Secrets:
   - Click 🔑 icon in left sidebar
   - Add secret: `FRED_API_KEY` = `your_key_here`

**Optional: Comtrade API Key**
- Free tier: 500 records/request (usually sufficient)
- Premium: https://comtrade.un.org/

### 4. Configure Data Collection

Edit the `COLLECTION_CONFIG` cell:
```python
COLLECTION_CONFIG = {
    'pairs': ['gbpusd', 'eurusd'],  # Add more pairs
    'start_date': '2020-01-01',
    'end_date': '2023-12-31',
    'collect_histdata': True,
    'collect_fundamentals': True,
    'collect_gdelt': False,  # Slow! Only if needed
}
```

### 5. Run Data Collection

Execute cells in order:
- **Step 2.2:** Collect HistData (price data)
- **Step 2.3:** Collect fundamental data (FRED, Comtrade, ECB)
- **Step 2.4:** (Optional) Collect GDELT sentiment

**Estimated time:**
- HistData: 5-10 min per pair per year
- Fundamentals: 1-2 min per pair
- GDELT: 30-60 min per pair per year (optional)

### 6. Prepare Datasets

Run **Step 3.1** to:
- Merge data sources
- Engineer features (SMA, RSI, Bollinger bands, etc.)
- Apply intrinsic time transformation (optional)
- Create train/val/test splits

### 7. Train Models

Configure training in **Step 4.1**:
```python
TRAINING_CONFIG = {
    'pair': 'gbpusd',
    'epochs': 50,
    'batch_size': 64,
    'training_type': 'supervised',  # or 'multitask', 'rl'
}
```

Then run **Step 4.2** (supervised), **4.3** (multitask), or **4.4** (RL).

---

## 💾 Storage Strategy

### Why Google Drive?

**Problem:** Codespace/Colab local storage is limited (50-100 GB) and **temporary**
- Data deleted when session ends
- Cannot store multi-year tick data
- Wastes time re-downloading data

**Solution:** Google Drive provides:
- **Persistent storage** (survives session restarts)
- **15 GB free** (enough for 2-3 pairs with 3 years data)
- **Automatic sync** (access from any Colab session)
- **Shareable** (collaborate with team)

### Storage Estimates

| Data Type | Size per Pair | Notes |
|-----------|---------------|-------|
| HistData (tick) | ~500 MB/year | Largest component |
| Prepared dataset | ~100-200 MB | After feature engineering |
| Fundamentals | ~5-10 MB | Economic indicators |
| GDELT sentiment | ~50-100 MB/year | Optional |
| Model checkpoint | ~50-100 MB | One model |

**Example:**
- GBPUSD + EURUSD (2020-2023)
- HistData: 2 pairs × 3 years × 500 MB = 3 GB
- Prepared: 2 × 200 MB = 400 MB
- Fundamentals: 2 × 10 MB = 20 MB
- Models: 5 × 100 MB = 500 MB
- **Total:** ~4 GB (fits in free tier)

### Expanding Storage

If you exceed 15 GB free tier:
1. **Google One:** $1.99/month for 100 GB
2. **Selective collection:** Focus on specific pairs/years
3. **Archive old data:** Move to external storage
4. **Compression:** GZIP prepared datasets

---

## 🎯 Workflow Examples

### Example 1: Quick Start (1 pair, 1 year)

```python
# Step 2.1: Minimal config
COLLECTION_CONFIG = {
    'pairs': ['gbpusd'],
    'start_date': '2023-01-01',
    'end_date': '2023-12-31',
    'collect_histdata': True,
    'collect_fundamentals': True,
    'collect_gdelt': False,
}

# Run: Step 2.2 → 2.3 → 3.1 → 4.2
# Time: ~20 minutes
# Storage: ~600 MB
```

### Example 2: Multi-Pair Research (3 years)

```python
COLLECTION_CONFIG = {
    'pairs': ['gbpusd', 'eurusd', 'usdjpy'],
    'start_date': '2020-01-01',
    'end_date': '2023-12-31',
    'collect_histdata': True,
    'collect_fundamentals': True,
}

# Run: Step 2.2 → 2.3 → 3.1 → 4.2
# Time: ~1 hour
# Storage: ~5 GB
```

### Example 3: Full Pipeline with Sentiment

```python
COLLECTION_CONFIG = {
    'pairs': ['gbpusd'],
    'start_date': '2022-01-01',
    'end_date': '2022-12-31',
    'collect_histdata': True,
    'collect_fundamentals': True,
    'collect_gdelt': True,  # Adds sentiment
    'include_sentiment': True,
}

# Run: All steps in order
# Time: ~2 hours (GDELT is slow)
# Storage: ~1 GB
```

---

## 🔧 Troubleshooting

### "RuntimeError: No space left on device"

**Cause:** Data being saved to local Colab disk instead of Drive

**Fix:** Check paths in config:
```python
# WRONG (local disk)
output = Path('/content/data')

# CORRECT (Google Drive)
output = DRIVE_ROOT / 'data' / 'prepared'
```

### "ModuleNotFoundError: No module named 'X'"

**Cause:** Dependencies not installed

**Fix:** Re-run Step 1.3 (Install Dependencies)

### "FRED API request failed"

**Cause:** API key not set or invalid

**Fix:**
1. Verify key is correct: https://fred.stlouisfed.org/
2. Check Colab Secrets (🔑 icon)
3. Or set manually: `os.environ['FRED_API_KEY'] = 'your_key'`

### "CUDA out of memory"

**Cause:** Batch size too large for GPU

**Fix:** Reduce batch size:
```python
TRAINING_CONFIG['batch_size'] = 32  # Try 32, then 16 if needed
```

### "HistData download timeout"

**Cause:** HistData.com is slow or overloaded

**Fix:**
1. Manual download: http://www.histdata.com/download-free-forex-data/
2. Upload CSVs to: `MyDrive/Sequence/data/raw/histdata/{pair}/`
3. Skip to Step 3.1 (Prepare Datasets)

### "Session disconnected"

**Cause:** Colab free tier has idle timeouts

**Prevention:**
- Run long tasks overnight
- Use Colab Pro for longer sessions
- Save checkpoints frequently (automatic in training)

**Recovery:**
1. Re-run Step 1.1 (Mount Drive)
2. Re-run Step 1.2 (Clone Repo)
3. Continue from where you left off (data is on Drive!)

---

## 💡 Best Practices

### 1. Incremental Data Collection

Don't collect everything at once:
```python
# Start small
'pairs': ['gbpusd'],
'start_date': '2023-01-01',
'end_date': '2023-06-30',

# Expand after verifying setup
'pairs': ['gbpusd', 'eurusd', 'usdjpy'],
'start_date': '2020-01-01',
'end_date': '2023-12-31',
```

### 2. Test Training First

Before collecting years of data:
1. Prepare 1 month of data
2. Train for 5 epochs
3. Verify pipeline works
4. Then collect full dataset

### 3. Regular Backups

Run Step 5.3 after each major milestone:
- After data collection
- After training completes
- Before experimenting with new configs

### 4. Monitor Storage

Run Step 5.1 frequently to check Drive usage:
```python
# Shows size of each directory
!du -sh {DRIVE_ROOT}/*
```

### 5. Clean Up

Remove unnecessary data to save space:
```python
# Delete raw HistData after preparation
!rm -rf {DRIVE_ROOT}/data/raw/histdata/{pair}

# Keep only best checkpoints
!rm {DRIVE_ROOT}/models/*_epoch_*.pt  # Keep only best_model.pt
```

---

## 📊 Expected Results

After completing the notebook, you should have:

**Data:**
- ✅ Raw price data on Drive
- ✅ Fundamental economic indicators
- ✅ Feature-engineered datasets ready for training

**Models:**
- ✅ Trained CNN-LSTM-Attention model
- ✅ Checkpoints saved to Drive
- ✅ Training logs and metrics

**Persistence:**
- ✅ All data persists across Colab sessions
- ✅ Can resume training from checkpoints
- ✅ No need to re-download data

---

## 🔗 Related Resources

- **Main Documentation:** [/CLAUDE.md](../CLAUDE.md)
- **Fundamental Data Guide:** [/docs/NEW_DATA_SOURCES.md](../docs/NEW_DATA_SOURCES.md)
- **Codebase Structure:** See Architecture Overview in CLAUDE.md
- **API Documentation:**
  - FRED: https://fred.stlouisfed.org/docs/api/
  - Comtrade: https://comtrade.un.org/api/
  - HistData: http://www.histdata.com/

---

## 📝 Notes

### Security
- **Never commit API keys** to notebooks
- Use Colab Secrets (🔑) for sensitive data
- API keys stored in `MyDrive/Sequence/config/` are private to your Drive

### Performance
- **Colab Free Tier:** T4 GPU, ~12-15 GB RAM, 50 GB disk
- **Colab Pro:** Better GPUs (V100/A100), more RAM, longer sessions
- **Training time:** ~1 hour for 50 epochs on T4

### Limitations
- Free tier has daily GPU limits
- Sessions timeout after ~12 hours idle
- Background execution requires Colab Pro

---

**Last Updated:** 2026-01-12
**Notebook Version:** 1.0
**Requires:** Google Colab, Google Drive account
