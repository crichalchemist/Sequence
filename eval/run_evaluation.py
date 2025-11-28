"""
Evaluation entrypoint to score trained models on the test split.

Example:
  python eval/run_evaluation.py \\
    --pairs gbpusd \\
    --years 2023 \\
    --t-in 60 --t-out 10 \\
    --task-type classification \\
    --checkpoint-path models/best_model.pt
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import ModelConfig, PolicyConfig, SignalModelConfig
from data.prepare_dataset import process_pair
from eval.agent_eval import evaluate_model, evaluate_policy_agent
from models.agent_hybrid import build_model
from models.signal_policy import SignalModel, SignalPolicyAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test split.")
    parser.add_argument("--pairs", default="gbpusd", help="Comma-separated pair codes")
    parser.add_argument("--years", default=None, help="Comma-separated years to include (default: all available)")
    parser.add_argument("--input-root", default="output_central", help="Root containing Central-time zips")
    parser.add_argument("--t-in", type=int, default=120)
    parser.add_argument("--t-out", type=int, default=10)
    parser.add_argument("--task-type", choices=["classification", "regression"], default="classification")
    parser.add_argument("--flat-threshold", type=float, default=0.0001)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--checkpoint-path", default="models/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--signal-checkpoint-path", default=None, help="Optional path to signal checkpoint (format string {pair} supported)")
    parser.add_argument("--policy-checkpoint-path", default=None, help="Optional path to policy checkpoint (format string {pair} supported)")
    parser.add_argument("--use-policy", action="store_true", help="Load the execution policy head on top of the signal model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-risk", action="store_true", help="Disable risk manager gating during evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = [p.strip().lower() for p in args.pairs.split(",") if p.strip()]

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    results = {}
    risk_manager = None if args.disable_risk else RiskManager()
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

        test_loader = loaders["test"]
        num_features = next(iter(test_loader))[0].shape[-1]

        if args.use_policy:
            signal_path = (
                Path(args.signal_checkpoint_path.format(pair=pair))
                if args.signal_checkpoint_path
                else Path(f"models/signal_{pair}.pt")
            )
            policy_path = (
                Path(args.policy_checkpoint_path.format(pair=pair))
                if args.policy_checkpoint_path
                else Path(f"models/policy_{pair}.pt")
            )
            if not signal_path.exists() or not policy_path.exists():
                print(
                    f"[error] signal/policy checkpoint missing: {signal_path} or {policy_path}"
                )
                continue

            signal_cfg = SignalModelConfig(
                num_features=num_features,
                num_classes=3 if args.task_type == "classification" else None,
                output_dim=1,
            )
            signal_dim = SignalModel(signal_cfg).signal_dim
            policy_cfg = PolicyConfig(
                input_dim=signal_dim,
                num_actions=3 if args.task_type == "classification" else 2,
            )
            agent = SignalPolicyAgent.load(
                signal_cfg,
                policy_cfg,
                str(signal_path),
                str(policy_path),
                device,
            )
            metrics = evaluate_policy_agent(agent, test_loader, task_type=args.task_type)
            results[pair_name] = metrics
            print(f"[eval-policy] {pair_name}: {metrics}")
            continue

        model_cfg = ModelConfig(
            num_features=num_features,
            num_classes=3 if args.task_type == "classification" else None,
        )
        model = build_model(model_cfg, task_type=args.task_type).to(device)

        ckpt_path = Path(args.checkpoint_path)
        if not ckpt_path.exists():
            print(f"[error] checkpoint not found: {ckpt_path}")
            continue
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

        metrics = evaluate_model(
            model, test_loader, task_type=args.task_type, risk_manager=risk_manager
        )
        results[pair_name] = metrics
        print(f"[eval] {pair_name}: {metrics}")

    return results


if __name__ == "__main__":
    main()
