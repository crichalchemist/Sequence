# Sequence – FX Forecasting Toolkit

A concise guide to get the repository up and running, covering data preparation, model training, evaluation, and
optional LLM finetuning.

## Quick‑start

1. **Install** – `python -m pip install -r requirements.txt` (Python 3.10+).
2. **Place data** – unzip HistData files under `output_central/<pair>/` (e.g. `output_central/gbpusd/2023.zip`).
3. **Prepare** – `python data/prepare_dataset.py --pairs gbpusd --t‑in 120 --t‑out 10 --task-type classification`.
4. **Train** – `python train/run_training.py --pairs gbpusd --epochs 3 --learning-rate 1e-3`.
5. **Evaluate** – `python eval/run_evaluation.py --pairs gbpusd --checkpoint-path models/gbpusd_best_model.pt`.

## Optional features

* **Intrinsic‑time bars** – directional‑change conversion: add `--intrinsic-time --dc-threshold-up 0.0005` to the
  prepare command.
* **Sentiment enrichment** – download GDELT GKG files with `--run-gdelt-download`; override the endpoint via
  `--gdelt-mirror` or `--gdelt-base-url`.
* **LLM finetuning** – resources for NovaSky models are in `train/`; see the script docstrings for 4‑bit LoRA vs.
  full‑precision options.

## Robustness & fault tolerance

* **Check‑pointing** – training writes checkpoints to `models/`; resume with `--resume-from-checkpoint <folder>`.
* **Idempotent pipelines** – existing checkpoints skip retraining; use `--force-train` to force a fresh run.

## Repository layout

* `data/` – raw loaders, intrinsic‑time conversion, and dataset builder.
* `features/` – pure technical‑indicator functions.
* `models/` – hybrid CNN + LSTM + Attention (`agent_hybrid.py`) and multi‑task variant.
* `train/` – training loops for signal and policy agents.
* `eval/` – evaluation utilities.
* `export/` – ONNX export helper.
* `utils/` – CLI wrappers and auxiliary helpers.

## Research context

We draw on a variety of finance‑focused studies (FX news impact, modern algorithmic trading, liquidity‑aware execution,
deep learning for FX, and reinforcement‑learning agents). See the original PDFs in the repository for details.

## References

* HistData – minute‑bar zip archives.
* GDELT – `http://data.gdeltproject.org/gdeltv2/`.
* NovaSky – local `Sky‑T1‑32B‑Flash` checkpoint.
