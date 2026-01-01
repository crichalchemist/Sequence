"""
RL-based execution policy training using advantage actor-critic (A2C).

This trains the ExecutionPolicy from models/signal_policy.py to learn optimal
execution decisions: when to act, which action to take (BUY/SELL/HOLD), and 
sizing constraints based on market state.

Example:
  python train/agent_train_rl.py \\
    --pairs gbpusd \\
    --t-in 120 \\
    --t-out 10 \\
    --batch-size 32 \\
    --epochs 10 \\
    --learning-rate 1e-4 \\
    --entropy-coeff 0.01 \\
    --value-loss-coeff 0.5 \\
    --gae-gamma 0.99 \\
    --gae-lambda 0.95 \\
    --checkpoint-path models/execution_policy.pt
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.prepare_dataset import prepare_data
from models.signal_policy import ExecutionPolicy, PolicyConfig
from utils.logger import get_logger
from utils.seed import set_seed
from utils.tracing import get_tracer, init_tracing_from_config

logger = get_logger(__name__)


@dataclass
class RLConfig:
    """RL hyperparameters for policy training."""
    entropy_coeff: float = 0.01          # Entropy regularization weight
    value_loss_coeff: float = 0.5        # Value network loss weight
    gae_gamma: float = 0.99              # Discount factor
    gae_lambda: float = 0.95             # GAE lambda for advantage smoothing
    clip_norm: float = 1.0               # Gradient clipping norm
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    device: str = "cpu"


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    GAE provides low-variance, smooth advantage estimates for policy gradient training.
    
    Args:
        rewards: Shape (T,) - rewards at each timestep
        values: Shape (T,) - value estimates at each timestep
        gamma: Discount factor
        lambda_: GAE smoothing parameter (0: high bias, low variance; 1: low bias, high variance)
    
    Returns:
        advantages: Shape (T,) - GAE advantages
        returns: Shape (T,) - discounted cumulative rewards
    """
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0.0

    # Backward pass for GAE computation
    for t in reversed(range(T)):
        # Temporal difference (TD) error
        if t == T - 1:
            next_value = 0.0  # Bootstrap: assume terminal state has value 0
        else:
            next_value = values[t + 1]

        td_error = rewards[t] + gamma * next_value - values[t]

        # GAE accumulation
        gae = td_error + gamma * lambda_ * gae
        advantages[t] = gae

    returns = advantages + values

    # Normalize advantages for stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def train_epoch(
    policy: ExecutionPolicy,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    rl_cfg: RLConfig,
    device: str,
) -> dict:
    """
    Train policy for one epoch using A2C updates.
    
    Returns:
        metrics: dict with 'policy_loss', 'value_loss', 'entropy', 'total_loss'
    """
    policy.train()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_batches = 0

    for batch in train_loader:
        if len(batch) == 2:
            x, y = batch
            x, y = x.to(device), y.to(device)
        else:
            # Extended batch with rewards/values for GAE
            x, y, rewards, values = batch
            x, y = x.to(device), y.to(device)
            rewards = rewards.to(device)
            values = values.to(device)

        # Forward pass: get policy logits and value
        policy_logits, value_pred = policy(x)  # logits: (B, 3), value: (B, 1)

        # Policy gradient loss: cross-entropy for action classification
        policy_loss = nn.functional.cross_entropy(policy_logits, y)

        # Value loss: MSE between predicted and target value
        value_loss = nn.functional.mse_loss(value_pred.squeeze(), rewards)

        # Entropy regularization: encourage exploration
        action_probs = torch.softmax(policy_logits, dim=-1)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1).mean()

        # Combined loss
        loss = (
            policy_loss
            + rl_cfg.value_loss_coeff * value_loss
            - rl_cfg.entropy_coeff * entropy
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), rl_cfg.clip_norm)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()
        num_batches += 1

    return {
        "policy_loss": total_policy_loss / num_batches,
        "value_loss": total_value_loss / num_batches,
        "entropy": total_entropy / num_batches,
        "total_loss": (total_policy_loss + rl_cfg.value_loss_coeff * total_value_loss) / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train RL execution policy using advantage actor-critic."
    )
    parser.add_argument(
        "--pairs",
        default="gbpusd",
        help="Comma-separated pair codes (e.g., gbpusd,eurusd)",
    )
    parser.add_argument("--t-in", type=int, default=120, help="Lookback window")
    parser.add_argument("--t-out", type=int, default=10, help="Forecast horizon")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--entropy-coeff", type=float, default=0.01, help="Entropy regularization")
    parser.add_argument("--value-loss-coeff", type=float, default=0.5, help="Value loss weight")
    parser.add_argument("--gae-gamma", type=float, default=0.99, help="GAE discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE smoothing")
    parser.add_argument(
        "--checkpoint-path",
        default="models/execution_policy.pt",
        help="Path to save best checkpoint",
    )
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--tracing-enabled",
        action="store_true",
        help="Enable OpenTelemetry tracing",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    # Initialize tracing if enabled
    tracer = None
    if args.tracing_enabled:
        tracing_cfg = {
            "service_name": "rl_training",
            "enabled": True,
            "jaeger_endpoint": "http://localhost:6831",
        }
        init_tracing_from_config(tracing_cfg)
        tracer = get_tracer(__name__)

    logger.info(f"Starting RL policy training with config:\n{args}")

    device = torch.device(args.device)

    # Load data
    with tracer.start_as_current_span("data_loading") if tracer else open('/dev/null'):
        try:
            pairs = [p.strip() for p in args.pairs.split(",")]
            logger.info(f"Loading data for pairs: {pairs}")

            data_dict = {}
            for pair in pairs:
                data_dict[pair] = prepare_data(
                    pair=pair,
                    t_in=args.t_in,
                    t_out=args.t_out,
                    task_type="classification",
                )
            logger.info(f"Loaded {len(data_dict)} pair datasets")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    # Combine all training data
    all_x_train = []
    all_y_train = []

    for pair, data in data_dict.items():
        x_train = torch.FloatTensor(data["x_train"])
        y_train = torch.LongTensor(data["y_train"])
        all_x_train.append(x_train)
        all_y_train.append(y_train)
        logger.info(f"{pair}: {x_train.shape[0]} train samples")

    x_train = torch.cat(all_x_train, dim=0)
    y_train = torch.cat(all_y_train, dim=0)

    logger.info(f"Combined training set: {x_train.shape}")

    # Create DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Initialize policy
    policy_cfg = PolicyConfig(
        input_size=x_train.shape[1],
        hidden_size=128,
        num_actions=3,  # BUY, SELL, HOLD
        dropout=0.2,
    )
    policy = ExecutionPolicy(policy_cfg).to(device)

    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    rl_cfg = RLConfig(
        entropy_coeff=args.entropy_coeff,
        value_loss_coeff=args.value_loss_coeff,
        gae_gamma=args.gae_gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
    )

    logger.info(f"RL Config:\n{rl_cfg}")

    best_loss = float("inf")
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Training loop
    with tracer.start_as_current_span("training") if tracer else open('/dev/null'):
        for epoch in range(1, args.epochs + 1):
            epoch_span = tracer.start_span(f"epoch_{epoch}") if tracer else None
            if epoch_span:
                epoch_span.__enter__()

            try:
                metrics = train_epoch(policy, optimizer, train_loader, rl_cfg, device)
                scheduler.step()

                logger.info(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f} | "
                    f"Entropy: {metrics['entropy']:.4f} | "
                    f"Total Loss: {metrics['total_loss']:.4f}"
                )

                if metrics["total_loss"] < best_loss:
                    best_loss = metrics["total_loss"]
                    torch.save(policy.state_dict(), checkpoint_path)
                    logger.info(f"✅ Saved checkpoint to {checkpoint_path}")

            finally:
                if epoch_span:
                    epoch_span.__exit__(None, None, None)

    logger.info(f"✅ RL policy training complete. Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
