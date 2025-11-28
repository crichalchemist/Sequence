# Plain-English Guide (with citations)
This repository trains and evaluates a foreign-exchange forecasting model on minute-level price data, with optional news sentiment. It also provides finetuning utilities for larger language models (NovaSky family) on finance-specific corpora.

## What you do (step by step)
1) Install dependencies: `python3 -m pip install -r requirements.txt`
2) Stage price data: place HistData zips under `output_central/PAIR/*.zip` (e.g., `output_central/gbpusd/2023.zip`). [HistData]
3) Train and test:
   - Train: `python utils/run_training_pipeline.py --pairs gbpusd --t-in 120 --t-out 10 --epochs 3`
   - Test: `python eval/run_evaluation.py --pairs gbpusd --checkpoint-path models/gbpusd_best_model.pt`
4) Optional sentiment: add `--run-gdelt-download` to pull GDELT GKG files before training. [Leetaru & Schrodt 2013]
5) Optional mini-game: when prompted during training, press `y` to launch; quit with `Q` (may slow training).

## Finetuning NovaSky models (LLMs)
- Resource-friendly (recommended on limited VRAM): `python train/run_finetune_novasky_lora.py --model-name-or-path /path/to/NovaSky-AI/Sky-T1-32B-Flash --output-dir models/novasky-lora --use-4bit --bf16 --per-device-train-batch-size 1 --gradient-accumulation-steps 32 --max-length 1024`
- Full finetune (heavier): `python train/run_finetune_novasky.py --model-name-or-path /path/to/NovaSky-AI/Sky-T1-32B-Flash --output-dir models/novasky-full --per-device-train-batch-size 1 --gradient-accumulation-steps 32 --max-length 1024`
- Finetune datasets (Hugging Face): FinanceInc/auditor_sentiment; yale-nlp/FinanceMath; PatronusAI/financebench; Josephgflowers/Finance-Instruct-500k.
- Base model: NovaSky-AI/Sky-T1-32B-Flash (local). [NovaSky 2025]

## Fault tolerance
- Frequent checkpoints; resume with `--resume-from-checkpoint <folder>`.
- Training pipeline skips pairs with existing checkpoints; override with `--force-train` to retrain.

## Components
- `data/`: price loaders and the GDELT downloader.
- `features/`: indicator/feature construction.
- `train/`: training loops and finetune runners.
- `eval/`: evaluation utilities.
- `utils/`: pipeline runner, mini-game, helpers.
- `models/agent_hybrid.py`: LSTM + CNN + Attention hybrid used for price forecasting (our LST/ATTN baseline).

## Research context (PDFs consulted)
- Applying news to FX: *Applying News-Based Trading to the FX Market* (applyingnews_forex.pdf).
- Modern algorithmic trading: *Modern Algorithmic Trading* (modernalgotrading.pdf).
- Liquidity and liquidation: *Liquidating FX Positions* (liquidatingforex.pdf).
- Execution: *Optimal Execution for Day Trading* (optimal execution day trading.pdf).
- Deep learning for FX: *Deep Learning for Forex* (forexdeeplearning.pdf); *Asynchronous Multi-Asset Learning* (deeplearning_asyncmulti.pdf); *Reinforcement Learning Agent for FX* (reinforcementlearning_agent.pdf).

These inform potential future work:
- Enrich sentiment: combine GDELT with economic calendars or headline embeddings.
- Multi-asset/async learning: explore asynchronous or multi-asset training for allocation and hedging.
- Execution-aware objectives: add slippage/impact-aware losses and evaluate under liquidity constraints.

## References
- [HistData] HistData minute bars (user-provided zips per pair).
- [Leetaru & Schrodt 2013] Leetaru, K., & Schrodt, P. A. (2013). GDELT: Global Database of Events, Language, and Tone, 1979–present. (GDELT 2.1 GKG endpoint: http://data.gdeltproject.org/gdeltv2/)
- [NovaSky 2025] NovaSky Team. (2025). Sky-T1-32B-Flash (local checkpoint) and SkyThought models.
- Hugging Face datasets: FinanceInc/auditor_sentiment; yale-nlp/FinanceMath; PatronusAI/financebench; Josephgflowers/Finance-Instruct-500k.
