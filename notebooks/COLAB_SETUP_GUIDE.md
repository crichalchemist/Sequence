# Google Colab Setup Guide for Sequence Trading System

This guide shows you how to adapt the existing `full_training_pipeline.ipynb` notebook to run on Google Colab.

## Prerequisites

1. **Upload Repository**: Zip the entire Sequence repository and upload `Sequence.zip` to your Google Drive (MyDrive root)
2. **Enable GPU**: In Colab, go to Runtime → Change runtime type → Hardware accelerator → GPU (T4 recommended for free tier)

## Step-by-Step Notebook Modifications

### 1. Add Google Drive Mount (New Cell at Beginning)

```python
from google.colab import drive
drive.mount('/content/drive')
print("✓ Google Drive mounted!")
```

### 2. Add Repository Extraction (New Cell)

```python
import zipfile
from pathlib import Path

zip_path = Path('/content/drive/MyDrive/Sequence.zip')
extract_to = Path('/content/Sequence')

if not zip_path.exists():
    print(f"❌ ERROR: {zip_path} not found")
    print("Upload Sequence.zip to your Google Drive root (MyDrive)")
else:
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('/content')

    # Handle possible directory naming
    extracted = [d for d in Path('/content').iterdir() if d.is_dir() and 'Sequence' in d.name]
    if extracted and extracted[0] != extract_to:
        extracted[0].rename(extract_to)

    print("✓ Repository extracted!")
```

### 3. Fix Path Configuration (Update Existing Cell #2)

**Replace** the existing "Setup and Imports" cell with:

```python
import sys
from pathlib import Path

# UPDATED FOR COLAB
ROOT = Path('/content/Sequence')  # Changed from /Volumes/Containers/Sequence
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'run'))

# Create necessary directories
(ROOT / 'data' / 'data').mkdir(parents=True, exist_ok=True)
(ROOT / 'data' / 'raw').mkdir(parents=True, exist_ok=True)
(ROOT / 'models' / 'checkpoints').mkdir(parents=True, exist_ok=True)

print(f"✓ Root: {ROOT}")
```

### 4. Update Requirements Installation (Update Cell #0)

```python
# Install PyTorch with CUDA support
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
!pip install -q numpy pandas matplotlib seaborn scikit-learn tqdm
!pip install -q transformers backtesting ta histdata

print("✓ All requirements installed!")
```

### 5. **ADD DATA COLLECTION CELL** (New - This is the missing piece!)

Insert this NEW cell **before** the "Data Preparation" section:

```python
# ============================================================================
# DATA COLLECTION FROM HISTDATA
# ============================================================================
from histdata.api import download_hist_data
from pathlib import Path

# Configuration (update these as needed)
PAIRS_TO_DOWNLOAD = ['gbpusd', 'eurusd']  # Start with 2-3 pairs
YEARS_TO_DOWNLOAD = ['2023', '2024']

print("📥 Downloading Historical FX Data from HistData.com")
print("="*70)
print(f"Pairs: {', '.join([p.upper() for p in PAIRS_TO_DOWNLOAD])}")
print(f"Years: {', '.join(YEARS_TO_DOWNLOAD)}")
print("="*70 + "\n")

raw_data_dir = ROOT / 'data' / 'raw'
raw_data_dir.mkdir(parents=True, exist_ok=True)

download_stats = {'successful': 0, 'failed': 0}

for pair in PAIRS_TO_DOWNLOAD:
    print(f"\n{'─'*70}")
    print(f"📊 Pair: {pair.upper()}")
    print(f"{'─'*70}")

    pair_dir = raw_data_dir / pair
    pair_dir.mkdir(parents=True, exist_ok=True)

    for year in YEARS_TO_DOWNLOAD:
        try:
            print(f"  {year}... ", end='', flush=True)

            # Try downloading full year first
            try:
                result = download_hist_data(
                    year=int(year),
                    pair=pair,
                    output_directory=str(pair_dir),
                    verbose=False
                )
                print(f"✓ {result}")
                download_stats['successful'] += 1

            except AssertionError:
                # Download month-by-month if full year not available
                print("(month-by-month)")
                months_ok = 0
                for month in range(1, 13):
                    try:
                        download_hist_data(
                            year=int(year),
                            month=month,
                            pair=pair,
                            output_directory=str(pair_dir),
                            verbose=False
                        )
                        months_ok += 1
                        download_stats['successful'] += 1
                    except:
                        pass
                print(f"    ✓ {months_ok} months downloaded")

        except Exception as e:
            print(f"✗ {str(e)[:50]}")
            download_stats['failed'] += 1

print(f"\n{'='*70}")
print(f"✓ Successful: {download_stats['successful']}")
print(f"✗ Failed: {download_stats['failed']}")
print(f"✓ Data collection complete!")
print(f"{'='*70}")
```

### 6. Update Data Preparation Cell

**Modify** the data preparation loop to use `PAIRS_TO_DOWNLOAD`:

```python
import subprocess

print("Starting data preparation pipeline...\\n")

for pair in PAIRS_TO_DOWNLOAD:  # Changed from PAIRS_TO_COLLECT
    print(f"\\nProcessing {pair.upper()}...")

    cmd = [
        'python', str(ROOT / 'data' / 'prepare_dataset.py'),
        '--pairs', pair,
        '--t-in', str(T_IN),
        '--t-out', str(T_OUT),
        '--task-type', TASK_TYPE,
    ]

    # Add optional flags
    if USE_INTRINSIC_TIME:
        cmd.append('--intrinsic-time')
    if INCLUDE_SENTIMENT:
        cmd.append('--include-sentiment')

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

    if result.returncode == 0:
        print(f"  ✓ {pair.upper()} prepared")
    else:
        print(f"  ✗ Failed: {result.stderr}")

print("\\n✓ Data preparation complete!")
```

### 7. Update Training Loop

**Change** `PAIRS` variable to `PAIRS_TO_DOWNLOAD`:

```python
for idx, pair in enumerate(PAIRS_TO_DOWNLOAD, 1):  # Changed from PAIRS
    print(f"\\n[{idx}/{len(PAIRS_TO_DOWNLOAD)}] Processing {pair.upper()}...")
    # ... rest of training code
```

### 8. Add Checkpoint Backup to Drive (New Final Cell)

```python
# Save checkpoints to Google Drive for persistence
import shutil

drive_checkpoint_dir = Path('/content/drive/MyDrive/Sequence_Models')
drive_checkpoint_dir.mkdir(parents=True, exist_ok=True)

print("Backing up checkpoints to Google Drive...\\n")

checkpoint_dir = ROOT / 'models' / 'checkpoints'
for checkpoint_file in checkpoint_dir.glob('*.pt'):
    shutil.copy2(checkpoint_file, drive_checkpoint_dir / checkpoint_file.name)
    print(f"✓ {checkpoint_file.name}")

print(f"\\n✓ Checkpoints saved to: {drive_checkpoint_dir}")
```

## Updated Configuration Section

Add this at the top of your configuration cell:

```python
# =============================================================================
# GOOGLE COLAB CONFIGURATION
# =============================================================================
# Start small for free tier limits (T4 GPU has ~15GB RAM)
PAIRS_TO_DOWNLOAD = ['gbpusd', 'eurusd']  # Limit to 2-3 pairs initially
YEARS_TO_DOWNLOAD = ['2023', '2024']      # 2 years of data

# Data settings
T_IN = 120
T_OUT = 10
TASK_TYPE = 'classification'
USE_INTRINSIC_TIME = False  # Set True for better performance (slower)
INCLUDE_SENTIMENT = False   # Set True to add GDELT sentiment (much slower)

# Training settings
TRAINING_MODE = "supervised"
SUPERVISED_CONFIG = {
    "epochs": 20,  # Reduced for Colab free tier time limits
    "batch_size": 64,
    "learning_rate": 1e-3,
    "use_amp": True,  # Enable for faster GPU training
    # ... rest of config
}
```

## Complete Cell Order

Your notebook should have cells in this order:

1. **Google Drive Mount** (NEW)
2. **Extract Repository** (NEW)
3. **Setup Paths** (MODIFIED - use `/content/Sequence`)
4. **Install Dependencies** (MODIFIED - add `histdata`)
5. **Configuration** (MODIFIED - add Colab-specific settings)
6. **GPU Check** (existing)
7. **Data Collection from HistData** (NEW - THE CRITICAL MISSING PIECE!)
8. **Data Preparation** (MODIFIED - use downloaded data)
9. **Setup and Imports** (existing)
10. **Helper Functions** (existing)
11. **Training Functions** (existing)
12. **Run Training** (MODIFIED - use correct pair list)
13. **Results & Visualization** (existing)
14. **Backup to Drive** (NEW)

## Tips for Google Colab

### Free Tier Limitations

- **12-hour session limit**: Long training runs may be interrupted
- **GPU timeout**: ~4-6 hours of continuous GPU use
- **Storage**: ~100GB disk space
- **RAM**: T4 GPU has ~15GB

### Optimization Strategies

1. **Start Small**: Train on 2-3 pairs first
2. **Enable AMP**: `use_amp: True` for 2x faster training
3. **Reduce Epochs**: Use 20-30 instead of 50 for free tier
4. **Save Frequently**: Backup checkpoints to Drive every 5-10 epochs
5. **Use Intrinsic Time Sparingly**: Adds preprocessing time but improves accuracy

### Handling Disconnections

If your Colab session disconnects:

1. Re-run the setup cells (Drive mount, extract, paths)
2. Skip data collection if files already exist in Drive
3. Resume training with `--resume-from-checkpoint` flag

## Quick Start Checklist

- [ ] Upload `Sequence.zip` to Google Drive MyDrive
- [ ] Create new Colab notebook
- [ ] Enable GPU runtime
- [ ] Add Google Drive mount cell
- [ ] Add repository extraction cell
- [ ] Update ROOT path to `/content/Sequence`
- [ ] Add `histdata` to pip install
- [ ] **Add data collection cell (critical!)**
- [ ] Update configuration for Colab
- [ ] Update all path references
- [ ] Add Drive backup cell
- [ ] Test with 1 pair first

## Troubleshooting

### "Sequence.zip not found"
- Verify file is in `/content/drive/MyDrive/Sequence.zip`
- Check Drive mount was successful
- Try refreshing Drive files

### "No module named 'config'"
- Verify `sys.path.insert(0, str(ROOT / 'run'))` was executed
- Check ROOT points to `/content/Sequence`

### "Out of memory" errors
- Reduce `batch_size` from 64 to 32
- Reduce number of pairs
- Disable `use_amp` (counter-intuitive but can help with RAM)

### HistData download fails
- Some pairs/years may not be available
- Downloads may be rate-limited
- Try month-by-month download (automatic fallback)

## Next Steps

After successful training on Colab:

1. Download checkpoints from Drive to local machine
2. Run evaluation on additional test data
3. Try ensemble methods with TimesFM
4. Experiment with RL training (A3C/SAC)
5. Enable intrinsic time for better performance

---

**Questions?** Check the main CLAUDE.md file for architecture details and common commands.
