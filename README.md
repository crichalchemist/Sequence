# Sequence – Deep Learning Framework for FX Market Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Sequence** is a comprehensive deep learning toolkit for foreign exchange (FX) market forecasting and algorithmic trading. The framework combines state-of-the-art neural architectures (CNN-LSTM-Attention hybrids), intrinsic time representations, sentiment analysis, and reinforcement learning for end-to-end trading strategy development.

## Overview

This repository implements a multi-modal approach to FX prediction integrating:

- **Hybrid Neural Architecture**: Deep CNN-LSTM networks with multi-head attention mechanisms for temporal pattern recognition
- **Intrinsic Time Representation**: Directional-change based time transformation for enhanced market structure capture [1]
- **Multi-Task Learning**: Joint prediction of price movements, volatility, and market regime
- **Sentiment Integration**: GDELT news event processing with FinBERT sentiment analysis
- **Reinforcement Learning**: A3C agents for execution policy optimization with backtesting integration
- **Production Pipeline**: Unified workflow supporting data acquisition, preprocessing, supervised learning, and RL training

## Key Features

- ✅ **End-to-End Pipeline**: Single command execution from data download to trained models
- ✅ **Intrinsic Time Bars**: Directional-change based time series transformation
- ✅ **Backtesting Integration**: Deterministic historical replay using backtesting.py
- ✅ **Multi-Task Architecture**: Simultaneous prediction of multiple market properties
- ✅ **Distributed Training**: Multi-GPU support with automatic mixed precision
- ✅ **Sentiment Enrichment**: Real-time GDELT news processing
- ✅ **RL Execution**: A3C policy learning for optimal order execution
- ✅ **Checkpoint Management**: Automatic model checkpointing and resume capability

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/crichalchemist/Sequence.git
cd Sequence

# Install dependencies (Python 3.10+)
python -m pip install -r requirements.txt
```

### Data Preparation

```bash
# Place HistData CSV files under output_central/<pair>/
# Example structure: output_central/gbpusd/2023.zip

# Prepare dataset with intrinsic time transformation
python data/prepare_dataset.py \
  --pairs gbpusd \
  --t-in 120 \
  --t-out 10 \
  --task-type classification \
  --intrinsic-time \
  --dc-threshold-up 0.0005
```

### Training

#### Supervised Learning

```bash
# Train hybrid CNN-LSTM-Attention model
python train/run_training.py \
  --pairs gbpusd \
  --epochs 50 \
  --learning-rate 1e-3 \
  --batch-size 64
```

#### Multi-Task Learning

```bash
# Train multi-task model (price + volatility + regime)
python train/run_training_multitask.py \
  --pairs gbpusd \
  --epochs 50 \
  --batch-size 64
```

#### Reinforcement Learning (A3C)

```bash
# Train RL execution policy with backtesting
python rl/run_a3c_training.py \
  --pair gbpusd \
  --env-mode backtesting \
  --historical-data data/data/gbpusd/gbpusd_prepared.csv \
  --num-workers 8 \
  --total-steps 1000000
```

### Unified Pipeline

```bash
# Complete workflow: download → prepare → train → evaluate → RL training
python utils/run_training_pipeline.py \
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
```

### Ensemble Evaluation with TimesFM

```bash
# Benchmark against Google's TimesFM foundation model
python eval/ensemble_timesfm.py \
  --pairs gbpusd \
  --years 2023 \
  --t-in 120 \
  --t-out 10 \
  --checkpoint-root models \
  --device cuda
```

## Architecture

### Data Pipeline Flow

```
Raw Data Sources
├── HistData OHLCV (1-minute bars)
├── GDELT News Events (Global Knowledge Graph)
└── TimesFM (Google foundation model - evaluation only)
                    ↓
Feature Engineering
├── Technical Indicators (SMA, EMA, RSI, Bollinger, ATR, etc.)
├── Intrinsic Time (Directional-change transformation)
├── FinBERT Sentiment (GDELT → FinBERT-tone → sentiment scores)
└── Feature Normalization
                    ↓
Model Training
├── Supervised Learning (CNN-LSTM-Attention)
├── Multi-Task Learning (Price + Volatility + Regime)
└── Reinforcement Learning (A3C execution policy)
                    ↓
Evaluation & Deployment
├── Backtesting (deterministic historical replay)
├── TimesFM Ensemble (benchmark vs foundation model)
└── Production Inference
```

### Neural Network Design

The core model (`models/agent_hybrid.py`) implements a hybrid architecture:

1. **Feature Extraction**: Multi-scale 1D CNN layers capture local patterns
2. **Temporal Modeling**: Bidirectional LSTM networks model long-term dependencies
3. **Attention Mechanism**: Multi-head self-attention for important feature weighting
4. **Output Heads**: Task-specific prediction layers (classification/regression)

### Intrinsic Time Representation

Traditional time-based sampling is replaced with directional-change events [1], providing:
- Scale-invariant market structure representation
- Reduced noise from low-activity periods
- Enhanced signal-to-noise ratio for pattern recognition

### Sentiment Integration Pipeline

Multi-stage news sentiment processing from GDELT Global Knowledge Graph:
1. **GDELT Download**: Retrieve Global Knowledge Graph event streams
2. **Event Filtering**: Extract FX-relevant news using entity/theme filtering  
3. **FinBERT Analysis**: Generate sentiment scores using FinBERT-tone model
4. **Feature Alignment**: Aggregate and align sentiment features with OHLCV price data
5. **Model Integration**: Concatenate sentiment features to technical indicators for training

The sentiment pipeline enriches price-based features with market psychology signals, enabling the model to learn correlations between news sentiment and price movements.

### Reinforcement Learning

A3C (Asynchronous Advantage Actor-Critic) agents learn optimal execution policies:
- **Simulated Mode**: Stochastic retail execution with realistic spread/slippage
- **Backtesting Mode**: Deterministic historical replay for reproducible experiments
- **Reward Function**: Risk-adjusted PnL with transaction cost penalties

### TimesFM Ensemble Forecasting

Post-training ensemble evaluation combines model predictions with Google's TimesFM foundation model:
1. **Trained Model**: CNN-LSTM-Attention predictions on normalized features
2. **TimesFM Forecasting**: Pre-trained foundation model predictions on raw price windows
3. **Ensemble**: Mean-weighted combination of both predictions
4. **Evaluation**: RMSE/MAE metrics on test set for model validation

TimesFM integration is used during evaluation (not training) to benchmark model performance against state-of-the-art foundation models.

## Repository Structure

```
Sequence/
├── data/               # Data loaders and preprocessing
│   ├── prepare_dataset.py
│   ├── intrinsic_time.py
│   └── iterable_dataset.py
├── features/           # Technical indicators and feature engineering
│   ├── agent_features.py
│   └── agent_sentiment.py
├── models/             # Neural network architectures
│   ├── agent_hybrid.py         # CNN-LSTM-Attention hybrid
│   ├── agent_multitask.py      # Multi-task variant
│   └── regime_encoder.py       # Market regime classifier
├── train/              # Training scripts
│   ├── run_training.py
│   └── run_training_multitask.py
├── rl/                 # Reinforcement learning
│   ├── run_a3c_training.py
│   └── agents/
├── eval/               # Evaluation utilities
│   ├── run_evaluation.py
│   └── agent_eval.py
├── execution/          # Trading environments
│   ├── backtesting_env.py
│   └── simulated_retail_env.py
├── utils/              # Pipeline orchestration
│   └── run_training_pipeline.py
└── gdelt/              # GDELT news processing
    ├── downloader.py
    └── feature_builder.py
```

## Advanced Usage

### Sentiment Enrichment

```bash
# Enable GDELT news download and FinBERT sentiment analysis
python data/prepare_dataset.py \
  --pairs gbpusd \
  --t-in 120 \
  --t-out 10 \
  --include-sentiment

# Or integrate into unified pipeline
python utils/run_training_pipeline.py \
  --pairs gbpusd \
  --run-gdelt-download \
  --include-sentiment \
  --gdelt-mirror https://custom-mirror.com  # Optional custom endpoint
```

### Custom Intrinsic Time Thresholds

```bash
# Adjust directional-change thresholds for different pairs
python data/prepare_dataset.py \
  --pairs gbpusd \
  --intrinsic-time \
  --dc-threshold-up 0.0008 \
  --dc-threshold-down 0.0008
```

### Multi-GPU Training

```bash
# Automatic multi-GPU detection with distributed data parallel
python train/run_training.py \
  --pairs gbpusd \
  --epochs 50 \
  --batch-size 128  # Will be split across GPUs
```

### Checkpoint Resume

```bash
# Resume training from checkpoint
python train/run_training.py \
  --pairs gbpusd \
  --resume-from-checkpoint models/checkpoint_epoch_25/
```

## Research Foundation

This framework is built upon established research in financial machine learning and algorithmic trading:

### Deep Learning for FX Markets

The hybrid CNN-LSTM-Attention architecture draws on modern deep learning approaches for financial time series [2, 3]. The multi-scale convolutional layers capture local patterns while LSTMs model long-term temporal dependencies.

### Intrinsic Time & Directional Change

Intrinsic time representation based on directional-change events provides scale-invariant market structure characterization [1], particularly effective for high-frequency FX data.

### News Sentiment & Market Impact

Integration of GDELT news sentiment follows research on news-driven FX movements [4], using FinBERT for domain-specific sentiment extraction.

### Optimal Execution & RL

The A3C execution policy is inspired by optimal execution research [5, 6] and modern algorithmic trading strategies [7], learning to minimize transaction costs while achieving target positions.

## Data Sources

- **Historical Price Data**: [HistData](https://www.histdata.com/) - Minute-level OHLCV data for major FX pairs
- **News Events**: [GDELT Project](http://data.gdeltproject.org/gdeltv2/) - Global Knowledge Graph for real-time event data
- **Sentiment Analysis**: [FinBERT-tone](https://huggingface.co/ProsusAI/finbert) - Financial domain sentiment classification
- **Foundation Model**: [TimesFM](https://github.com/google-research/timesfm) - Google's pre-trained time series forecasting model (evaluation only)

## References

[1] Glattfelder, J. B., Dupuis, A., & Olsen, R. B. (2011). Patterns in high-frequency FX data: Discovery of 12 empirical scaling laws. *Quantitative Finance*, 11(4), 599-614.

[2] Dixon, M., Klabjan, D., & Bang, J. H. (2017). Classification-based financial markets prediction using deep neural networks. *Algorithmic Finance*, 6(3-4), 67-77.

[3] Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654-669.

[4] Sinha, N. R., & Tewari, A. (2020). Applying news analytics to financial markets: An empirical study. *Journal of Banking and Financial Technology*, 4(1), 59-73.

[5] Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3, 5-39.

[6] Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.

[7] Chan, E. P. (2021). *Quantitative Trading: How to Build Your Own Algorithmic Trading Business* (2nd ed.). Wiley.

[8] Das, A., Kong, W., Leach, A., Mathur, S., Sen, R., & Yu, R. (2024). A decoder-only foundation model for time-series forecasting. In *Proceedings of the 41st International Conference on Machine Learning* (ICML 2024).

## Documentation

For detailed usage guides and integration examples, see:

- [Backtesting Integration Guide](docs/guides/BACKTESTING_INTEGRATION_GUIDE.md) - Comprehensive guide to RL training with backtesting.py
- [Architecture & API Reference](docs/api/ARCHITECTURE_API_REFERENCE.md) - Technical deep-dive into model architecture
- [Tracing Implementation Guide](docs/guides/TRACING_IMPLEMENTATION.md) - Observability and debugging with tracing
- [Research Evaluation](docs/research/RESEARCH_EVALUATION.md) - Analysis of foundational research papers
- [Troubleshooting FAQ](TROUBLESHOOTING_FAQ.md) - Common issues and solutions

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{sequence_fx_toolkit,
  title = {Sequence: Deep Learning Framework for FX Market Prediction},
  author = {crichalchemist},
  year = {2024},
  url = {https://github.com/crichalchemist/Sequence}
}
```

## Acknowledgments

- HistData.com for historical FX data
- GDELT Project for global event data
- PyTorch team for the deep learning framework
- backtesting.py authors for the backtesting infrastructure
