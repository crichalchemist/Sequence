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
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from config.arg_parser import (  # noqa: E402
    add_amp_args,
    add_auxiliary_head_weights,
    add_data_preparation_args,
    add_dataloader_args,
    add_feature_engineering_args,
    add_intrinsic_time_args,
    add_risk_args,
    add_rl_training_args,
    add_training_args,
)
from config.config import (  # noqa: E402
    PolicyConfig,
    RLTrainingConfig,
    SignalModelConfig,
    TrainingConfig,
)
from data.prepare_dataset import process_pair  # noqa: E402
from models.signal_policy import ExecutionPolicy, SignalModel  # noqa: E402
from train.core.agent_train import (  # noqa: E402
    pretrain_signal_model,
    train_execution_policy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run training for all pairs.")
    add_data_preparation_args(parser)
    add_feature_engineering_args(parser)
    add_intrinsic_time_args(parser)
    add_training_args(parser)
    add_dataloader_args(parser)
    add_amp_args(parser)
    parser.add_argument("--checkpoint-path", default="models/best_model.pt")
    add_risk_args(parser)
    add_auxiliary_head_weights(parser)
    parser.add_argument("--signal-checkpoint-path", default="models/signal_{pair}.pt")
    parser.add_argument("--policy-checkpoint-path", default="models/policy_{pair}.pt")
    add_rl_training_args(parser)
    return parser.parse_args()


def main():
    # Initialize tracing for observability
    try:
        from utils.tracing import setup_tracing
        setup_tracing(
            service_name="sequence-training",
            otlp_endpoint="http://localhost:4318",
            environment="development"
        )
    except ImportError:
        print("[warn] OpenTelemetry not available; running without tracing")
    except Exception as e:
        print(f"[warn] Failed to initialize tracing: {e}; continuing without tracing")

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
            lookahead_window = args.lookahead_window
            top_k = args.top_k
            predict_sell_now = args.predict_sell_now
            target_type = args.task_type
            flat_threshold = args.flat_threshold
            train_ratio = args.train_ratio
            val_ratio = args.val_ratio
            batch_size = args.batch_size
            feature_groups = args.feature_groups
            exclude_feature_groups = args.exclude_feature_groups
            sma_windows = args.sma_windows
            ema_windows = args.ema_windows
            rsi_window = args.rsi_window
            bollinger_window = args.bollinger_window
            bollinger_num_std = args.bollinger_num_std
            atr_window = args.atr_window
            short_vol_window = args.short_vol_window
            long_vol_window = args.long_vol_window
            spread_windows = args.spread_windows
            imbalance_smoothing = args.imbalance_smoothing
            intrinsic_time = args.intrinsic_time
            dc_threshold_up = args.dc_threshold_up
            dc_threshold_down = args.dc_threshold_down

        try:
            pair_name, loaders = process_pair(pair, PrepArgs, batch_size=args.batch_size)
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
            lookahead_window=args.lookahead_window,
            top_k_predictions=args.top_k,
            predict_sell_now=args.predict_sell_now,
            output_dim=1,
        )
        pretrain_cfg = TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.pretrain_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            max_return_weight=args.max_return_weight,
            topk_return_weight=args.topk_return_weight,
            topk_price_weight=args.topk_price_weight,
            sell_now_weight=args.sell_now_weight,
            checkpoint_path=str(signal_ckpt),
        )
        pretrain_cfg.risk.enabled = not args.disable_risk

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
