# Sequence Complete Platform Integration Guide

## 🎉 System Now Fully Integrated

Your Sequence FX Intelligence Platform now includes complete end-to-end functionality for **data collection, preprocessing, training, backtesting, and MQL5 integration**.

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                      │
│  (http://localhost:8504) - Matrix-Themed Control Center    │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │ DATA PIPE   │    │  TRAINING   │    │ BACKTESTING │
   │  MANAGER    │    │  MANAGER    │    │  MANAGER    │
   └─────────────┘    └─────────────┘    └─────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
   ┌─────────────────────┐              ┌──────────────────────┐
   │   SQLITE DATABASE   │              │  MQL5 REST API       │
   │  (Multi-layer)      │              │  (Flask Server)      │
   └─────────────────────┘              └──────────────────────┘
        │                                           │
   ┌────┴─────────────┐                      ┌──────┴──────┐
   │ Pipeline Jobs    │                      │ MT5 Ticks   │
   │ Training Jobs    │                      │ Signals     │
   │ Backtest Results │                      │ Backtest    │
   │ GPU Stats        │                      │ Results     │
   └──────────────────┘                      └─────────────┘
                                                      │
                                                      ▼
                                            ┌──────────────────┐
                                            │  MetaTrader 5    │
                                            │  Live Trading &  │
                                            │  Backtesting    │
                                            └──────────────────┘
```

---

## 🔌 MQL5 REST API Server

### Start the API Server

```bash
cd /home/crichalchemist/Sequence
python mql5/api_server.py
```

Server runs on: **http://localhost:5000**

### API Documentation

Visit: **http://localhost:5000/api/v1/docs**

### Available Endpoints

#### Health Check
```
GET /api/v1/health
```
Returns system status, versions, and component health.

#### Live Data from MT5
```
POST /api/v1/live_data/tick
{
    "symbol": "GBPUSD",
    "bid": 1.2534,
    "ask": 1.2536,
    "volume": 1000000,
    "timestamp": 1701956400.0
}
```

#### Retrieve Live Data
```
GET /api/v1/live_data/GBPUSD?limit=100
```

#### Trading Signals
```
POST /api/v1/signals/create
{
    "symbol": "GBPUSD",
    "signal_type": "BUY",
    "confidence": 0.85,
    "entry_price": 1.2534,
    "stop_loss": 1.2500,
    "take_profit": 1.2600
}
```

#### Get Pending Signals
```
GET /api/v1/signals/pending
```

#### Mark Signal Sent
```
POST /api/v1/signals/123/sent
```

#### Import Backtest Results
```
POST /api/v1/backtest/import
{
    "strategy_name": "MA_Crossover",
    "symbol": "GBPUSD",
    "timeframe": "1H",
    "start_date": "2025-01-01",
    "end_date": "2025-12-06",
    "total_return": 15.5,
    "sharpe_ratio": 1.45,
    "win_rate": 65.2,
    "max_drawdown": -8.3,
    "trades_count": 125
}
```

#### Get Backtest Comparison
```
GET /api/v1/backtest/comparison/GBPUSD?limit=10
```

#### Export to CSV
```
POST /api/v1/backtest/export
{
    "symbol": "GBPUSD",
    "output_path": "output_central/backtest_results.csv"
}
```

---

## 🎓 Data Pipeline Controller

### Unified Data Collection

```python
from data.pipeline_controller import controller, DataConfig

# Configure data collection
config = DataConfig(
    data_sources=['GDELT', 'YFinance'],
    symbols=['GBPUSD', 'EURUSD'],
    countries=['US', 'EU'],
    start_date='2025-01-01',
    end_date='2025-12-06',
    resolution='daily',
    preprocessing={
        'normalize': True,
        'normalization_method': 'zscore',
        'engineer_features': True
    }
)

# Collect data
data, collection_id = controller.collect_data(config)

# Preprocess
processed, prep_id = controller.preprocess(data, collection_id, config.preprocessing)

# Validate
validation_result, val_id = controller.validate(processed, collection_id)

# Check status
status = controller.get_pipeline_status()
```

### Data Sources Supported

- **GDELT**: Global event data with sentiment analysis
- **YFinance**: FX and stock market data
- **HistData**: Historical FX data (implementation ready)

### Preprocessing Features

- Z-score and MinMax normalization
- Moving average engineering
- Volatility calculation
- Missing value handling (forward/backward fill)

### Validation Rules

- Minimum 100 rows
- Max 5% missing values
- Price bounds: 0.001 - 1,000,000
- Volume bounds: 0 - 1,000,000,000
- 3-sigma outlier detection

---

## 🤖 Training Queue Manager

### Submit Training Job

```python
from train.training_manager import manager, TrainingJob

job = TrainingJob(
    job_id="train_hybrid_20251206",
    model_name="Hybrid CNN-LSTM",
    dataset_path="output_central/gbpusd_dataset.csv",
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    validation_split=0.2,
    early_stopping=True,
    patience=10,
    priority=2
)

# Submit to queue
manager.submit_job(job)

# Monitor status
status = manager.get_job_status(job.job_id)
print(f"Status: {status['status']}")
print(f"Epoch: {status['current_epoch']}/{job.epochs}")
```

### GPU Monitoring

```python
# Get GPU status
gpu_status = manager.gpu_monitor.get_status()

# Returns:
# {
#     "has_gpu": True,
#     "device_count": 2,
#     "devices": [
#         {
#             "gpu_id": 0,
#             "name": "Tesla V100",
#             "memory_allocated_mb": 2048,
#             "memory_reserved_mb": 4096,
#             "utilization_percent": 50.0
#         }
#     ],
#     "cpu_percent": 25.3,
#     "memory_percent": 45.6
# }
```

### Queue Management

```python
# Get queue status
queue_status = manager.get_queue_status()

# Get running jobs
running_jobs = manager.get_running_jobs()

# Automatic resource allocation based on:
# - GPU memory available
# - CPU utilization
# - Concurrent job limit (default: 2)
```

---

## 📊 Backtesting with Comparison

### Run Backtest with backtesting.py

```python
from execution.backtest_manager import manager
import pandas as pd

# Load data
data = pd.read_csv('data.csv', index_col=0, parse_dates=True)

# Define custom strategy class
from backtesting import Strategy

class MyStrategy(Strategy):
    def init(self):
        pass
    
    def next(self):
        pass

# Run backtest
result = manager.run_backtest(
    data=data,
    strategy_class=MyStrategy,
    strategy_name="MA_Crossover",
    symbol="GBPUSD",
    cash=10000,
    commission=0.001
)

# Save result
manager.save_result(
    run_id="backtest_001",
    strategy_name="MA_Crossover",
    symbol="GBPUSD",
    timeframe="1H",
    start_date="2025-01-01",
    end_date="2025-12-06",
    cash=10000,
    commission=0.001,
    result=result
)
```

### Compare Strategies

```python
# Compare two backtests
comparison = manager.compare_strategies("backtest_001", "backtest_002")

# Returns:
# {
#     "strategy_1": "MA_Crossover",
#     "strategy_2": "RSI_Bands",
#     "winner": "MA_Crossover",
#     "metrics": {
#         "return": {"strategy_1": 15.5, "strategy_2": 12.3, "delta": 3.2},
#         "sharpe_ratio": {"strategy_1": 1.45, "strategy_2": 1.22, "delta": 0.23},
#         "max_drawdown": {"strategy_1": -8.3, "strategy_2": -10.5, "delta": 2.2}
#     }
# }

# Export comparison
manager.export_comparison_csv("backtest_001", "backtest_002", "comparison.csv")

# Get portfolio statistics
stats = manager.get_portfolio_stats()
```

---

## 🎨 Dashboard Integration

The Matrix-themed Streamlit dashboard now includes new tabs:

### 📥 Data Pipeline Tab
- Select data sources (GDELT, YFinance, HistData)
- Configure preprocessing options
- Monitor validation results
- Download processed datasets

### 🎓 Training Hub Tab
- Submit training jobs
- Monitor GPU utilization
- Track training progress in real-time
- Compare model performance

### 📊 Backtesting Tab
- Select strategy and parameters
- Run manual backtests
- Compare strategy performance
- Export results for MQL5

### 🔌 MQL5 Bridge Tab
- View API connection status
- Monitor live data feed
- Manage trading signals
- Import MT5 backtest results
- Compare Sequence vs MT5 results

---

## 💾 Database Structure

### MQL5 Data (`output_central/mql5_data.db`)
- **backtest_results**: Imported MT5 backtest results
- **live_ticks**: Live price data from MT5
- **trading_signals**: Generated signals for MT5

### Backtest Results (`output_central/backtest_results.db`)
- **backtest_runs**: Strategy backtest results
- **backtest_comparisons**: Strategy comparison data

### Training Jobs (`output_central/training_jobs.db`)
- **training_jobs**: Job queue and history
- **gpu_stats**: GPU utilization over time

### Data Pipeline (`output_central/data_pipeline.db`)
- **data_collections**: Data collection jobs
- **data_preprocessing**: Preprocessing history
- **data_validation**: Validation results
- **dataset_versions**: Dataset versioning

---

## 🚀 Complete Workflow Example

### 1. Collect Data
```bash
# Via Dashboard or API
# Data Pipeline Tab → Select sources → Execute
```

### 2. Preprocess & Validate
```bash
# Automatic in dashboard
# Check validation results
```

### 3. Train Models
```bash
# Training Hub Tab → Submit job → Monitor progress
```

### 4. Backtest Strategy
```bash
# Backtesting Tab → Run backtest → Get results
```

### 5. Deploy to MT5
```bash
# MQL5 Bridge → Start API server
# Configure MT5 EA to connect to http://localhost:5000
```

### 6. Live Trading
```bash
# MT5 sends live ticks to /api/v1/live_data/tick
# Sequence generates signals → /api/v1/signals/pending
# MT5 executes signals
```

### 7. Compare Results
```bash
# Backtesting Tab → Compare Sequence vs MT5 results
# Export CSV → Analyze in Excel
```

---

## 📝 MT5 Expert Advisor Integration

### Example MQL5 EA pseudocode

```mql5
// Connect to Sequence API
#define API_URL "http://localhost:5000/api/v1"

void OnStart() {
    // Send live tick
    SendTick("GBPUSD", Ask, Bid);
}

void SendTick(string symbol, double bid, double ask) {
    string json = StringFormat(
        "{\"symbol\":\"%s\",\"bid\":%.5f,\"ask\":%.5f,\"volume\":%d,\"timestamp\":%.0f}",
        symbol, bid, ask, Volume[0], TimeCurrent()
    );
    
    // POST to /api/v1/live_data/tick
    // Use WinHttpRequest or similar
}

void CheckSignals() {
    // GET /api/v1/signals/pending
    // Execute signals received from Sequence
}

void SendBacktestResults() {
    // POST /api/v1/backtest/import
    // Submit MT5 backtest results for comparison
}
```

---

## 🔧 Configuration Files

### Data Pipeline Config
Located in `data/pipeline_controller.py`

### Training Config
Located in `train/training_manager.py`

### Backtest Config
Located in `execution/backtest_manager.py`

### MQL5 API Config
Located in `mql5/api_server.py`

---

## 📚 Database Queries

### Get Latest Backtest Results
```sql
SELECT * FROM backtest_results 
ORDER BY import_timestamp DESC 
LIMIT 10;
```

### Get Running Training Jobs
```sql
SELECT * FROM training_jobs 
WHERE status = 'RUNNING';
```

### Get GPU Stats Over Time
```sql
SELECT * FROM gpu_stats 
ORDER BY timestamp DESC 
LIMIT 1000;
```

### Get Validation Issues
```sql
SELECT * FROM data_validation 
WHERE status = 'FAIL' 
ORDER BY created_at DESC;
```

---

## 🎯 Next Steps

1. **Start MQL5 API Server**
   ```bash
   python mql5/api_server.py
   ```

2. **Configure MT5 Expert Advisor**
   - Set API URL: `http://localhost:5000`
   - Enable live data transmission
   - Set signal generation interval

3. **Monitor Dashboard**
   - http://localhost:8504
   - Check all pipeline stages
   - Monitor training and backtesting

4. **Compare Results**
   - Sequence backtesting vs MT5 Strategy Tester
   - Live trading vs strategy expectations
   - Optimize parameters

---

## ✅ System Status

- ✅ **Data Pipeline**: Fully integrated with collection, preprocessing, validation
- ✅ **Training Manager**: Queue-based with GPU monitoring and resource management
- ✅ **Backtesting**: Comparison and storage with historical analysis
- ✅ **MQL5 REST API**: Complete endpoints for live data and signal exchange
- ✅ **Dashboard Integration**: All components connected to Streamlit UI
- ✅ **SQLite Databases**: Separate databases for each module (simplicity & comparison)

---

**Your Sequence Platform is now production-ready for full end-to-end AI-powered trading!**

