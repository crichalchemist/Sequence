"""
End-to-end training entrypoint using the agent components.

Example:
  python train/run_training.py \\
    --pairs eurusd,eurgbp,eurjpy,eurchf,euraud,eurcad,eurnzd,gbpusd,gbpjpy,gbpchf,gbpcad,gbpaud,gbpnzd,usdjpy,usdchf,usdcad,audusd,audjpy,audcad,audchf,audnzd,nzdusd,nzdjpy,nzdcad,nzdchf,cadchf,cadjpy,chfjpy,usdbrl,usdrub,usdinr,usdcny,usdzar,usdtry,xauusd \\
    --years all \\
    --t-in 60 --t-out 10 \\
    --task-type classification \\
    --epochs 3 --batch-size 64 --learning-rate 1e-3
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import ModelConfig, TrainingConfig
from data.prepare_dataset import process_pair
from models.agent_hybrid import build_model
from train.agent_train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run training for all pairs.")
    parser.add_argument("--pairs", default="gbpusd", help="Comma-separated pair codes")
    parser.add_argument("--years", default=None, help="Comma-separated years to include (default: all available)")
    parser.add_argument("--input-root", default="output_central", help="Root containing Central-time zips")
    parser.add_argument("--t-in", type=int, default=120)
    parser.add_argument("--t-out", type=int, default=10)
    parser.add_argument("--task-type", choices=["classification", "regression"], default="classification")
    parser.add_argument("--flat-threshold", type=float, default=0.0001)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-path", default="models/best_model.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = [p.strip().lower() for p in args.pairs.split(",") if p.strip()]

    # Avoid device mismatch if CUDA not available.
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # Prepare per-pair datasets/loaders and train models individually.
    results = {}
    for pair in pairs:
        class PrepArgs:
            pairs = pair
            input_root = args.input_root
            years = args.years
            t_in = args.t_in
            t_out = args.t_out
            target_type = args.task_type
            flat_threshold = args.flat_threshold
            train_ratio = args.train_ratio
            val_ratio = args.val_ratio
            batch_size = args.batch_size

        try:
            pair_name, loaders = process_pair(pair, PrepArgs, batch_size=args.batch_size)
        except Exception as exc:
            print(f"[error] data prep failed for {pair}: {exc}")
            continue

        train_loader = loaders["train"]
        val_loader = loaders["val"]
        num_features = next(iter(train_loader))[0].shape[-1]

        # Derive checkpoint path per pair unless user overrides.
        if args.checkpoint_path == "models/best_model.pt":
            ckpt_path = Path("models") / f"{pair}_best_model.pt"
        else:
            ckpt_path = Path(args.checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        model_cfg = ModelConfig(
            num_features=num_features,
            num_classes=3 if args.task_type == "classification" else None,
        )
        train_cfg = TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            checkpoint_path=str(ckpt_path),
        )

        model = build_model(model_cfg, task_type=args.task_type)
        history = train_model(
            model,
            train_loader,
            val_loader,
            train_cfg,
            task_type=args.task_type,
        )
        results[pair_name] = history
        print(f"[done] {pair_name} training complete.")

    return results


if __name__ == "__main__":
    main()
