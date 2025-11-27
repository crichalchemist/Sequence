# FutureFi

Time-series FX forecasting with hybrid CNN/LSTM/Attention models, multi-task heads, and optional sentiment features.

## Quick start
- Python 3.10+ recommended. Create a venv: `python3 -m venv .venv && source .venv/bin/activate`
- Install deps (includes editable TimesFM): `pip install -r requirements.txt`
- Data lives under `output_central/` (zipped HistData inputs expected there). Outputs and checkpoints go to `output/` and `models/`.
- Large model weights (FinBERT tone, TimesFM sources) are tracked with Git LFS; ensure LFS is installed before cloning/pulling.

## Data preparation
Prepare sliding-window datasets (time-ordered splits, normalized features):
```bash
python data/prepare_dataset.py --pairs gbpusd --t-in 120 --t-out 10 --task-type classification
```
For multi-task datasets:
```bash
python data/prepare_multitask_dataset.py --pairs gbpusd --t-in 120 --t-out 10
```
Inputs should be minute bars; place zipped HistData files under `output_central/` or adjust the script paths to your storage.

## Training
Single-task training:
```bash
python train/run_training.py --pairs gbpusd --t-in 120 --t-out 10 --epochs 3 --learning-rate 1e-3 --checkpoint-path models/gbpusd_best_model.pt
```
Multi-task training:
```bash
python train/run_training_multitask.py --pairs gbpusd --t-in 120 --t-out 10 --epochs 3 --checkpoint-path models/gbpusd_best_multitask.pt
```
Use short runs (`--epochs 1`) for quick smoke tests.

## Evaluation
Evaluate a trained single-task checkpoint:
```bash
python eval/run_evaluation.py --pairs gbpusd --checkpoint-path models/gbpusd_best_model.pt
```

## Export
Export to ONNX:
```bash
python - <<'PY'
from models.agent_hybrid import HybridModel
from export.agent_export import export_to_onnx, ExportConfig
import torch

model = HybridModel(...)  # load or construct your trained model
model.load_state_dict(torch.load("models/gbpusd_best_model.pt", map_location="cpu"))
example_input = torch.zeros(1, 120, model.input_size)
export_to_onnx(model, ExportConfig(onnx_path="models/hybrid.onnx"), example_input)
PY
```

## Sentiment scoring (optional)
`features/agent_sentiment.py` can attach FinBERT tone scores to news. The repo includes local weights under `models/finBERT-tone/`; loading uses the standard `transformers` pipeline.

## Project layout
- `config/` dataclass configs
- `features/agent_features.py` feature engineering (pure functions)
- `data/` loading, normalization, windowing, dataset builders
- `models/` architectures (CNN + LSTM + Attention; multi-head)
- `train/` training loops and runners; `eval/` mirrors for evaluation
- `export/` ONNX export helpers
- `output_central/` expected HistData zips; `output/` run artifacts/checkpoints

## Tips
- Keep time splits ordered; avoid shuffling across splits.
- If you change data prep, verify window counts and normalization stats printed by the scripts.
- ONNX validation: load with `onnxruntime` if available.
