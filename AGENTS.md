# Repository Guidelines

## Project Structure & Module Organization
- `config/` holds dataclass configs consumed by every agent.
- `features/agent_features.py` builds technical indicators; keep functions pure (no I/O).
- `data/agent_data.py` and `data/prepare_dataset.py` load minute bars, add features, and build sliding-window datasets plus DataLoaders.
- `models/` contains the CNN + LSTM + Attention architectures (`agent_hybrid.py`) and multi-head variant (`agent_multitask.py`).
- `train/` provides training loops (`agent_train.py`, `agent_train_multitask.py`) and CLI runners; `eval/` mirrors this for evaluation.
- `export/agent_export.py` handles ONNX export; `output_central/` is the expected location for zipped HistData inputs; `models/` stores checkpoints.

## Setup, Build, and Development Commands
- Python 3.10+; install deps in a virtualenv: `pip install -r requirements.txt` (includes editable `timesfm`).
- Prepare data (time-ordered splits, normalized features): `python data/prepare_dataset.py --pairs gbpusd --t-in 120 --t-out 10 --task-type classification`.
- Train single-task model: `python train/run_training.py --pairs gbpusd --t-in 120 --t-out 10 --epochs 3 --learning-rate 1e-3 --checkpoint-path models/gbpusd_best_model.pt`.
- Train multi-task model: `python train/run_training_multitask.py --pairs gbpusd --t-in 120 --t-out 10 --epochs 3 --checkpoint-path models/gbpusd_best_multitask.pt`.
- Evaluate on test split: `python eval/run_evaluation.py --pairs gbpusd --checkpoint-path models/gbpusd_best_model.pt`.
- Export ONNX (example): load the trained model and call `export.export_to_onnx(model, ExportConfig(onnx_path="models/hybrid.onnx"), example_input)`.

## Coding Style & Naming Conventions
- Python: 4-space indent, type hints, dataclasses for configs, snake_case for functions/vars, PascalCase for classes.
- Keep agent boundaries clean: feature computation stays in FeatureAgent, windowing/labels in DataAgent, architecture-only logic in ModelAgent, training/eval loops in their agents.
- Prefer pure functions and deterministic behavior (no shuffling across time splits); avoid embedding file I/O inside feature/model code.

## Testing Guidelines
- No formal test suite yet; smoke-test changes by running a short training epoch on a small pair (e.g., `--epochs 1 --pairs gbpusd`) and then `eval/run_evaluation.py` to confirm metrics compute.
- When altering data prep, ensure window counts and normalization stats printouts look plausible and that splits remain time-ordered.
- For export changes, verify ONNX creation and loadability with `onnxruntime` if available.

## Commit & Pull Request Guidelines
- Use concise, imperative commit messages (e.g., "Add attention dropout", "Fix dataset windowing"); keep related changes grouped per commit.
- PRs should describe the task, affected agents/files, configs used, and sample commands run (train/eval/export). Include any new arguments or defaults and note checkpoint/output paths touched.
