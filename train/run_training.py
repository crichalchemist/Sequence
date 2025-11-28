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

from config.config import (  # noqa: E402
    PolicyConfig,
    RLTrainingConfig,
    SignalModelConfig,
    TrainingConfig,
)
from data.prepare_dataset import process_pair  # noqa: E402
from models.signal_policy import ExecutionPolicy, SignalModel  # noqa: E402
from train.agent_train import (  # noqa: E402
    pretrain_signal_model,
    train_execution_policy,
)


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
    parser.add_argument("--signal-checkpoint-path", default="models/signal_{pair}.pt")
    parser.add_argument("--policy-checkpoint-path", default="models/policy_{pair}.pt")
    parser.add_argument("--pretrain-epochs", type=int, default=5, help="epochs for signal pretraining")
    parser.add_argument("--policy-epochs", type=int, default=5, help="epochs for execution policy training")
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--detach-signal", action="store_true", help="freeze signal encoder during policy training")
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

        try:
            pair_name, loaders = process_pair(pair, PrepArgs)
        except Exception as exc:
            print(f"[error] data prep failed for {pair}: {exc}")
            continue

        train_loader = loaders["train"]
        val_loader = loaders["val"]
        num_features = next(iter(train_loader))[0].shape[-1]

        signal_ckpt = Path(args.signal_checkpoint_path.format(pair=pair))
        policy_ckpt = Path(args.policy_checkpoint_path.format(pair=pair))
        signal_ckpt.parent.mkdir(parents=True, exist_ok=True)
        policy_ckpt.parent.mkdir(parents=True, exist_ok=True)

        signal_cfg = SignalModelConfig(
            num_features=num_features,
            num_classes=3 if args.task_type == "classification" else None,
            output_dim=1,
        )
        pretrain_cfg = TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.pretrain_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            checkpoint_path=str(signal_ckpt),
        )
        train_cfg.risk.enabled = not args.disable_risk

        signal_model = SignalModel(signal_cfg)
        signal_history = pretrain_signal_model(
            signal_model,
            train_loader,
            val_loader,
            pretrain_cfg,
            task_type=args.task_type,
        )

        policy_cfg = PolicyConfig(
            input_dim=signal_model.signal_dim,
            num_actions=3 if args.task_type == "classification" else 2,
        )
        rl_cfg = RLTrainingConfig(
            epochs=args.policy_epochs,
            learning_rate=args.learning_rate,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            detach_signal=args.detach_signal,
            checkpoint_path=str(policy_ckpt),
        )
        policy_head = ExecutionPolicy(policy_cfg)
        train_execution_policy(
            signal_model,
            policy_head,
            train_loader,
            rl_cfg,
            task_type=args.task_type,
        )

        results[pair_name] = {"signal_history": signal_history}
        print(f"[done] {pair_name} signal+policy training complete.")

    return results


if __name__ == "__main__":
    main()
