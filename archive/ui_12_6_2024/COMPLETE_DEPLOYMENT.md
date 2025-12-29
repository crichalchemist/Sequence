# 🏛️ Sequence FX Intelligence Platform - Complete Deployment Guide

## 🎉 System Status: FULLY OPERATIONAL

Your Sequence platform now includes **complete end-to-end AI trading infrastructure** with data collection, preprocessing, training, backtesting, and MetaTrader 5 integration.

---

## 🚀 Quick Start

### Single Command Launch
```bash
cd /home/crichalchemist/Sequence
./start_platform.sh
```

This starts:
- **MQL5 REST API Server** on `http://localhost:5000`
- **Streamlit Dashboard** on `http://localhost:8504`

### Manual Start

**Terminal 1 - MQL5 API Server:**
```bash
cd /home/crichalchemist/Sequence
python mql5/api_server.py
```

**Terminal 2 - Streamlit Dashboard:**
```bash
cd /home/crichalchemist/Sequence
streamlit run streamlit_matrix_app.py --server.port 8504
```

---

## 📊 System Architecture

### Five Core Modules

1. **Data Pipeline Controller** (`data/pipeline_controller.py`)
   - Multi-source data collection (GDELT, YFinance, HistData)
   - Preprocessing (normalization, feature engineering)
   - Validation (quality scoring, outlier detection)
   - SQLite storage for reproducibility

2. **Training Manager** (`train/training_manager.py`)
   - Job queue with priority scheduling
   - GPU monitoring and resource allocation
   - Automatic batch size adjustment
   - Real-time progress tracking

3. **Backtesting Manager** (`execution/backtest_manager.py`)
   - backtesting.py integration
   - Multi-strategy comparison
   - Side-by-side metrics analysis
   - Historical result storage

4. **MQL5 Bridge** (`mql5/bridge.py`)
   - Live tick data ingestion
   - Trading signal generation
   - Backtest result import
   - Database storage (SQLite)

5. **REST API Server** (`mql5/api_server.py`)
   - Flask-based REST API
   - JSON request/response format
   - CORS enabled for cross-origin requests
   - Complete documentation at `/api/v1/docs`

---

## 🔌 MQL5 Integration

### What is the REST API?

A simple HTTP-based communication protocol between your Streamlit system and MetaTrader 5:

```
MetaTrader 5 (Running on Windows/Mac)
         ↓ (HTTP POST)
  Send live price ticks
         ↓
  Sequence REST API (http://localhost:5000)
         ↑ (HTTP GET)
  Retrieve trading signals
         ↓
  Execute on your MT5 account
```

### Setting Up MT5 Connection

**In your MQL5 Expert Advisor:**

```mql5
#include <WinHttpRequest.mqh>

WinHttpRequest http;
string API_URL = "http://127.0.0.1:5000/api/v1";

void OnTick() {
    // Send live tick
    string json = StringFormat("{\"symbol\":\"GBPUSD\",\"bid\":%.5f,\"ask\":%.5f}",
                              Bid, Ask);
    http.Send(API_URL + "/live_data/tick", json);
    
    // Get signals
    string response = http.Receive(API_URL + "/signals/pending");
    // Process response...
}
```

### API Endpoints Quick Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/live_data/tick` | Send live tick from MT5 |
| GET | `/api/v1/live_data/<symbol>` | Get recent live data |
| POST | `/api/v1/signals/create` | Create trading signal |
| GET | `/api/v1/signals/pending` | Get pending signals |
| POST | `/api/v1/backtest/import` | Import MT5 backtest |
| GET | `/api/v1/backtest/comparison/<symbol>` | Compare backtests |
| GET | `/api/v1/training/gpu/status` | Check GPU usage |

---

## 📊 Dashboard Interface

### 6 Main Tabs

#### 1. 🔲 System Matrix
- Real-time system status
- Model performance overview
- Architecture visualization
- Component health diagnostics

#### 2. 📡 Data Streams  
- Multi-source data collection
- Real-time preprocessing
- Data quality validation
- Dataset versioning

#### 3. 🤖 Neural Networks
- Model performance metrics
- Training progress tracking
- Model comparison
- Architecture overview

#### 4. 💱 FX Analytics
- Currency pair analysis
- Trading signal generation
- Performance metrics
- Risk analysis

#### 5. 🔬 Research Lab
- Active experiments
- Research findings
- Correlation analysis
- Feature importance

#### 6. ⚙️ Infrastructure
- GPU utilization
- Service status
- System logs
- Resource monitoring

---

## 🎓 Complete Workflow

### Step 1: Collect Data
```python
from data.pipeline_controller import controller, DataConfig

config = DataConfig(
    data_sources=['GDELT', 'YFinance'],
    symbols=['GBPUSD'],
    countries=['US', 'EU'],
    start_date='2025-01-01',
    end_date='2025-12-06',
    resolution='daily',
    preprocessing={'normalize': True, 'engineer_features': True}
)

data, collection_id = controller.collect_data(config)
```

### Step 2: Preprocess
```python
processed, prep_id = controller.preprocess(data, collection_id, config.preprocessing)
validation, val_id = controller.validate(processed, collection_id)
print(f"Quality Score: {validation['quality_score']}")
```

### Step 3: Train Model
```python
from train.training_manager import manager, TrainingJob

job = TrainingJob(
    job_id="train_001",
    model_name="Hybrid CNN-LSTM",
    dataset_path="output_central/dataset.csv",
    epochs=100,
    batch_size=32,
    learning_rate=0.001
)

manager.submit_job(job)
status = manager.get_job_status(job.job_id)
```

### Step 4: Backtest
```python
from execution.backtest_manager import manager

result = manager.run_backtest(
    data=data,
    strategy_class=MyStrategy,
    strategy_name="MA_Crossover",
    symbol="GBPUSD"
)

manager.save_result("bt_001", "MA_Crossover", "GBPUSD", 
                   "1H", "2025-01-01", "2025-12-06", 
                   10000, 0.001, result)
```

### Step 5: Deploy to MT5
```bash
# Start API server
python mql5/api_server.py

# Configure MT5 EA with:
# - API_URL = "http://127.0.0.1:5000"
# - Connect to /api/v1/live_data/tick
# - Poll /api/v1/signals/pending for signals
```

### Step 6: Compare Results
```python
# Sequence vs MT5
comparison = manager.compare_strategies("seq_bt_001", "mt5_bt_001")

# Export for analysis
manager.export_comparison_csv("seq_bt_001", "mt5_bt_001", 
                             "sequence_vs_mt5.csv")
```

---

## 💾 Database Structure

### 4 SQLite Databases (Simplicity + Comparison)

**`output_central/mql5_data.db`** - MQL5 Integration
- `backtest_results` - Imported MT5 results
- `live_ticks` - Live price data
- `trading_signals` - Generated signals

**`output_central/backtest_results.db`** - Backtesting
- `backtest_runs` - Backtest history
- `backtest_comparisons` - Strategy comparisons

**`output_central/training_jobs.db`** - Training
- `training_jobs` - Job queue & history
- `gpu_stats` - GPU monitoring

**`output_central/data_pipeline.db`** - Data Pipeline
- `data_collections` - Collection jobs
- `data_preprocessing` - Preprocessing jobs
- `data_validation` - Validation results
- `dataset_versions` - Version history

---

## 🎯 Key Features

### ✅ Data Collection
- GDELT global events
- YFinance FX/stock data
- HistData historical data
- Automatic caching

### ✅ Preprocessing
- Z-score normalization
- MinMax scaling
- Moving average engineering
- Volatility calculation
- Missing value handling

### ✅ Validation
- Quality scoring (0-100)
- Outlier detection (3-sigma)
- Missing value checking
- Data bounds validation

### ✅ Training Management
- Priority queue scheduling
- GPU memory monitoring
- CPU utilization tracking
- Concurrent job limiting
- Automatic checkpointing

### ✅ Backtesting
- Multiple strategy support
- Parameter optimization
- Side-by-side comparison
- Historical tracking
- CSV export

### ✅ MQL5 Integration
- REST API (HTTP)
- JSON data format
- Live tick ingestion
- Signal distribution
- Result import/comparison

---

## 🔧 Configuration

### Data Pipeline
```python
# Min quality score for validation
validation_rules = {
    'min_rows': 100,
    'max_missing_percent': 0.05,
    'price_bounds': (0.001, 1000000),
    'volume_bounds': (0, 1000000000)
}
```

### Training Manager
```python
# Max concurrent jobs
max_concurrent_jobs = 2

# GPU monitoring enabled by default
gpu_monitor = GPUMonitor()
```

### API Server
```python
# API runs on port 5000
# CORS enabled for cross-origin requests
# Debug mode available for development
```

---

## 📈 Performance Metrics

### Expected Performance
- Data collection: 1-5 minutes per symbol/timeframe
- Preprocessing: < 1 second per 100K rows
- Validation: < 500ms per dataset
- Training: Depends on model size (hours to days)
- API response time: < 100ms

### Resource Requirements
- **CPU**: 4+ cores recommended
- **GPU**: NVIDIA/AMD optional (for training)
- **Memory**: 16GB+ recommended
- **Disk**: 100GB+ for datasets/models
- **Network**: For live data streaming

---

## 🐛 Troubleshooting

### API Server Won't Start
```bash
# Check if port 5000 is in use
lsof -i :5000

# Kill process on port 5000
kill -9 $(lsof -t -i :5000)
```

### Dashboard Won't Load
```bash
# Verify Python dependencies
pip list | grep streamlit

# Reinstall if needed
pip install --upgrade streamlit plotly
```

### GPU Not Detected
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show GPU name
```

### Database Locked
```bash
# Remove lockfiles
rm output_central/*.db-wal
rm output_central/*.db-shm
```

---

## 📚 Documentation

- **Full Integration Guide**: `PLATFORM_INTEGRATION_GUIDE.md`
- **Dashboard Deploy Summary**: `DASHBOARD_DEPLOYMENT_SUMMARY.md`
- **MCP Conversion Guide**: `MCP_CONVERSION_README.md`
- **Troubleshooting FAQ**: `TROUBLESHOOTING_FAQ.md`
- **Project README**: `README.md`

---

## 🚀 Next Steps

1. **Start Platform**
   ```bash
   ./start_platform.sh
   ```

2. **Access Dashboard**
   - Open http://localhost:8504
   - Explore System Matrix tab

3. **Configure MT5**
   - Download latest MT5 example EA
   - Set API URL: http://127.0.0.1:5000
   - Connect live account/backtest

4. **Run First Workflow**
   - Collect data (Data Streams tab)
   - Train model (Neural Networks tab)
   - Backtest strategy (Backtesting tab)
   - Deploy signals to MT5

5. **Monitor Results**
   - Track live data in MQL5 Bridge
   - Compare Sequence vs MT5 results
   - Optimize parameters

---

## ✨ Platform Highlights

✅ **End-to-End**: Data → Training → Backtesting → MT5  
✅ **Scalable**: GPU monitoring + resource management  
✅ **Comparable**: Cross-platform result comparison  
✅ **Simple**: REST API + JSON format  
✅ **Trackable**: SQLite databases for all components  
✅ **Visual**: Matrix-themed Streamlit dashboard  
✅ **Integrated**: Complete AI trading system  

---

## 🎊 You're Ready!

Your Sequence FX Intelligence Platform is fully operational with:

- 📊 **Complete data pipeline** (collection → preprocessing → validation)
- 🤖 **Advanced training management** (queue, GPU monitoring, comparison)
- 📈 **Professional backtesting** (strategy comparison, historical tracking)
- 🔌 **MQL5 REST API** (live data, signals, backtest import)
- 🎨 **Matrix-themed dashboard** (monitoring, control, visualization)

**Start exploring your AI-powered trading platform!**

```bash
./start_platform.sh
```

Access at:
- Dashboard: http://localhost:8504
- API: http://localhost:5000
- Docs: http://localhost:5000/api/v1/docs

