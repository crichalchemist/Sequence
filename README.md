# Plain-English Guide
We teach a computer to spot patterns in currency prices. It reads minute-by-minute price files, learns, and saves a model you can test or reuse. You can also add news tone data and even play a tiny game while it runs.

## What you do (step by step)
1) Install the tools: `python3 -m pip install -r requirements.txt`
2) Put price zips into `output_central/PAIR/*.zip` (for example, `output_central/gbpusd/2023.zip`).
3) Train and test:
   - Train: `python utils/run_training_pipeline.py --pairs gbpusd --t-in 120 --t-out 10 --epochs 3`
   - Test: `python eval/run_evaluation.py --pairs gbpusd --checkpoint-path models/gbpusd_best_model.pt`
4) Want news tone in the mix? Add `--run-gdelt-download` to the training command.
5) Bored during training? When asked, press `y` to play the mini-game (quit with `Q`). It can slow training a bit.

## Bigger models (NovaSky)
- Lighter on memory (recommended): `python train/run_finetune_novasky_lora.py --model-name-or-path /path/to/NovaSky-AI/Sky-T1-32B-Flash --output-dir models/novasky-lora --use-4bit --bf16 --per-device-train-batch-size 1 --gradient-accumulation-steps 32 --max-length 1024`
- Heavier (full finetune): `python train/run_finetune_novasky.py --model-name-or-path /path/to/NovaSky-AI/Sky-T1-32B-Flash --output-dir models/novasky-full --per-device-train-batch-size 1 --gradient-accumulation-steps 32 --max-length 1024`
- These use four finance datasets to teach the model: FinanceInc/auditor_sentiment, yale-nlp/FinanceMath, PatronusAI/financebench, Josephgflowers/Finance-Instruct-500k.

## If the machine shuts off
- Training saves checkpoints often. Restart with `--resume-from-checkpoint <folder>`.
- The pipeline skips pairs that already have a checkpoint. Add `--force-train` if you want to redo them.

## Where things live
- `data/` price loaders and the GDELT downloader.
- `features/` the math that turns prices into indicators.
- `train/` training loops and finetune scripts.
- `eval/` testing.
- `utils/` helpers (pipeline runner, mini-game).

## Sources
- Price data: you supply HistData minute bars (zipped) per pair.
- News tone: GDELT 2.1 GKG (http://data.gdeltproject.org/gdeltv2/).
- Finetune datasets: FinanceInc/auditor_sentiment, yale-nlp/FinanceMath, PatronusAI/financebench, Josephgflowers/Finance-Instruct-500k on Hugging Face.
- NovaSky model family: NovaSky-AI/Sky-T1-32B-Flash (local checkpoint).
