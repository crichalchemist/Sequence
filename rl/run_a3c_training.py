"""CLI runner for A3C RL agent training with simulated execution environments."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import ModelConfig  # noqa: E402
from execution.simulated_retail_env import SimulatedRetailExecutionEnv, ExecutionConfig  # noqa: E402
from rl.agents.a3c_agent import A3CAgent, A3CConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for A3C training."""
    parser = argparse.ArgumentParser(
        description="Train an A3C agent for FX execution with simulated retail environment."
    )
    
    # Environment and data args
    parser.add_argument("--pair", type=str, default="gbpusd", help="Currency pair (e.g., gbpusd, eurusd)")
    parser.add_argument("--initial-balance", type=float, default=10000.0, help="Initial account balance in USD")
    parser.add_argument(
        "--env-mode",
        type=str,
        default="simulated",
        choices=["simulated", "backtesting"],
        help="Environment mode: 'simulated' for stochastic retail execution, 'backtesting' for deterministic historical replay"
    )
    parser.add_argument(
        "--historical-data",
        type=str,
        default=None,
        help="Path to historical OHLCV CSV file (required for backtesting mode)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/data",
        help="Root directory for prepared data (used to auto-locate historical data)"
    )
    
    # Model architecture args
    parser.add_argument("--num-features", type=int, default=20, help="Number of input features")
    parser.add_argument("--hidden-size-lstm", type=int, default=64, help="LSTM hidden dimension")
    parser.add_argument("--num-layers-lstm", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--cnn-num-filters", type=int, default=32, help="Number of CNN filters")
    parser.add_argument("--cnn-kernel-size", type=int, default=3, help="CNN kernel size")
    parser.add_argument("--attention-dim", type=int, default=64, help="Attention hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # A3C training args
    parser.add_argument("--num-workers", type=int, default=4, help="Number of asynchronous workers")
    parser.add_argument("--total-steps", type=int, default=100_000, help="Total environment steps")
    parser.add_argument("--rollout-length", type=int, default=5, help="Steps per rollout before update")
    
    # Optimizer args
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")
    
    # Loss weighting args
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="Value loss scaling")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm")
    
    # RL hyperparams
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    
    # System args
    parser.add_argument("--device", default="cuda", help="Device for training (cpu or cuda)")
    parser.add_argument("--checkpoint-path", default="models/a3c_agent.pt", help="Checkpoint save path")
    parser.add_argument("--log-interval", type=int, default=1000, help="Logging interval (steps)")
    
    return parser.parse_args()


def main():
    """Main training loop for A3C agent."""
    args = parse_args()
    
    # Validate device
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Validate environment mode
    if args.env_mode == "backtesting":
        # Import backtesting environment (lazy to avoid hard dependency)
        try:
            from execution.backtesting_env import BacktestingRetailExecutionEnv  # noqa: F401
        except ImportError as exc:
            log.error(f"backtesting.py is required for --env-mode=backtesting")
            log.error(f"Install with: pip install backtesting>=0.3.2")
            sys.exit(1)
        
        # Locate historical data
        if args.historical_data:
            data_path = Path(args.historical_data)
        else:
            # Auto-locate from data_root
            data_path = Path(args.data_root) / args.pair / f"{args.pair}_prepared.csv"
            if not data_path.exists():
                # Try alternate locations
                alt_paths = [
                    Path(args.data_root) / f"{args.pair}.csv",
                    Path("data") / args.pair / f"{args.pair}.csv",
                    Path(f"{args.pair}.csv"),
                ]
                for alt in alt_paths:
                    if alt.exists():
                        data_path = alt
                        break
        
        if not data_path.exists():
            log.error(f"Historical data not found: {data_path}")
            log.error(f"Provide --historical-data or prepare data at {data_path}")
            sys.exit(1)

        log.info(f"[a3c] Using historical data: {data_path}")

    log.info(f"[a3c] Training configuration:")
    log.info(f"  - Pair: {args.pair}")
    log.info(f"  - Environment mode: {args.env_mode}")
    log.info(f"  - Workers: {args.num_workers}")
    log.info(f"  - Total steps: {args.total_steps}")
    log.info(f"  - Learning rate: {args.learning_rate}")
    log.info(f"  - Device: {device}")
    log.info(f"  - Checkpoint: {args.checkpoint_path}")
    
    # Create model config
    model_cfg = ModelConfig(
        num_features=args.num_features,
        hidden_size_lstm=args.hidden_size_lstm,
        num_layers_lstm=args.num_layers_lstm,
        cnn_num_filters=args.cnn_num_filters,
        cnn_kernel_size=args.cnn_kernel_size,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    )
    
    # Create A3C config
    a3c_cfg = A3CConfig(
        n_workers=args.num_workers,
        total_steps=args.total_steps,
        rollout_length=args.rollout_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        gamma=args.gamma,
        max_grad_norm=args.max_grad_norm,
        checkpoint_path=args.checkpoint_path,
        log_interval=args.log_interval,
        device=device,
    )
    
    # Create environment factory
    if args.env_mode == "simulated":
        # Stochastic retail simulation with spread, slippage, and realistic fills
        def make_env():
            exec_cfg = ExecutionConfig()
            return SimulatedRetailExecutionEnv(
                config=exec_cfg,
                pair=args.pair,
                initial_balance=args.initial_balance,
            )

        log.info(f"[a3c] Environment: SimulatedRetailExecutionEnv (stochastic)")
    
    else:  # backtesting mode
        # Deterministic historical replay with backtesting.py
        pass


from utils.logger import get_logger

log = get_logger(__name__)
        
        # Load historical data once (shared across workers)
log.info(f"[a3c] Loading historical OHLCV data from {data_path}...")
        price_df = pd.read_csv(data_path)
        
        # Ensure datetime column exists and is parsed
        if "datetime" in price_df.columns:
            price_df["datetime"] = pd.to_datetime(price_df["datetime"])
            price_df = price_df.set_index("datetime")
        
        # Validate required columns
        required_cols = {"open", "high", "low", "close"}
        available_cols = {c.lower() for c in price_df.columns}
        if not required_cols.issubset(available_cols):
            missing = required_cols - available_cols
            log.error(f"Missing required OHLCV columns: {missing}")
            log.error(f"Available columns: {list(price_df.columns)}")
            sys.exit(1)

log.info(f"[a3c] Loaded {len(price_df)} bars for backtesting")
        
        def make_env():
            exec_cfg = ExecutionConfig(initial_cash=args.initial_balance)
            return BacktestingRetailExecutionEnv(
                price_df=price_df.copy(),
                config=exec_cfg,
            )


log.info(f"[a3c] Environment: BacktestingRetailExecutionEnv (deterministic historical)")
    
    # Create and train agent
log.info(f"\n[a3c] Initializing agent...")
    agent = A3CAgent(
        model_cfg=model_cfg,
        a3c_cfg=a3c_cfg,
        action_dim=3,  # hold, buy, sell
        env_factory=make_env,
    )

log.info(f"[a3c] Starting training ({args.num_workers} workers)...")
    agent.train()

log.info(f"[a3c] Training complete! Checkpoint saved to: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
