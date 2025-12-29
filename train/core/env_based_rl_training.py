"""
Real RL training using execution environment with actual PnL rewards.

This module replaces the fake reward lookup tables with genuine reinforcement learning
from trading outcomes. The agent learns by interacting with SimulatedRetailExecutionEnv
and receiving rewards based on actual portfolio value changes.

Key differences from agent_train_rl.py:
- Uses execution environment (not static labels)
- Rewards come from actual PnL (not [-1, 0, 1] lookup table)
- Episodes generate trajectories through environment interaction
- Policy gradient uses real trading outcomes
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import PolicyConfig, RLTrainingConfig, SignalModelConfig
from execution.simulated_retail_env import (
    ExecutionConfig,
    OrderAction,
    SimulatedRetailExecutionEnv,
)
from models.policy import ExecutionPolicy, SignalModel
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger(__name__)


class ActionConverter:
    """Converts policy network outputs to OrderAction objects for the environment.

    Phase 3: Enhanced with risk-based position sizing capabilities.
    """

    def __init__(
            self,
            lot_size: float = 1.0,
            action_mode: str = "discrete",
            max_position: float = 10.0,
            risk_per_trade: float = 0.02,
            use_dynamic_sizing: bool = False,
    ):
        """
        Args:
            lot_size: Base size of each trading lot
            action_mode: "discrete" (buy/hold/sell) or "continuous" (position sizing)
            max_position: Maximum absolute inventory allowed (risk limit)
            risk_per_trade: Fraction of portfolio to risk per trade (for dynamic sizing)
            use_dynamic_sizing: If True, scale position size based on portfolio and volatility
        """
        self.lot_size = lot_size
        self.action_mode = action_mode
        self.max_position = max_position
        self.risk_per_trade = risk_per_trade
        self.use_dynamic_sizing = use_dynamic_sizing

    def policy_to_order(
            self, action_idx: int, mid_price: float, inventory: float, cash: float = 50_000.0
    ) -> OrderAction:
        """
        Convert discrete policy action to OrderAction with risk-based position sizing.

        Args:
            action_idx: 0=SELL, 1=HOLD, 2=BUY
            mid_price: Current market mid price
            inventory: Current position inventory
            cash: Available cash for position sizing

        Returns:
            OrderAction for the environment
        """
        if action_idx == 1:  # HOLD
            return OrderAction(action_type="hold", side="buy", size=0.0)

        side = "buy" if action_idx == 2 else "sell"

        # Phase 3: Calculate position size based on risk parameters
        size = self._calculate_position_size(side, inventory, mid_price, cash)

        return OrderAction(
            action_type="market",
            side=side,
            size=size,
        )

    def _calculate_position_size(
            self, side: str, inventory: float, mid_price: float, cash: float
    ) -> float:
        """
        Calculate position size based on risk management rules.

        This method implements position sizing logic that balances:
        - Risk per trade (using self.risk_per_trade)
        - Maximum position limits (self.max_position)
        - Current inventory (avoid excessive concentration)
        - Available cash

        Args:
            side: "buy" or "sell"
            inventory: Current position inventory (positive=long, negative=short)
            mid_price: Current market mid price
            cash: Available cash

        Returns:
            Position size in lots
        """
        # Phase 3: Dynamic position sizing based on portfolio value
        if self.use_dynamic_sizing:
            # Calculate portfolio value (cash + unrealized position value)
            portfolio_value = cash + (inventory * mid_price)

            # Risk a fixed percentage of portfolio per trade
            # This creates a Kelly-criterion-like approach where position size
            # scales with portfolio growth/shrinkage
            risk_amount = portfolio_value * self.risk_per_trade

            # Convert risk amount to position size in lots
            # Using mid_price as the risk basis (could also use ATR/volatility)
            size = risk_amount / mid_price

            # Round to lot_size increments
            size = max(self.lot_size, round(size / self.lot_size) * self.lot_size)
        else:
            # Fixed lot size (simpler approach)
            size = self.lot_size

        # Apply hard position limits (prevent catastrophic concentration)
        # This is a safety constraint, but the RL agent learns optimal sizing
        # through reward signals (PnL, Sharpe ratio, drawdown, etc.)
        if side == "buy":
            # Check if buying would exceed max position
            new_inventory = inventory + size
            if new_inventory > self.max_position:
                # Reduce size to stay within limit
                size = max(0.0, self.max_position - inventory)
        else:  # sell
            # Check if selling would exceed max short position
            new_inventory = inventory - size
            if new_inventory < -self.max_position:
                # Reduce size to stay within limit
                size = max(0.0, self.max_position + inventory)

        # Apply cash constraint (can't buy more than we can afford)
        if side == "buy":
            max_affordable = cash / mid_price
            size = min(size, max_affordable)

        # Ensure size is non-negative and rounded to lot increments
        size = max(0.0, size)
        size = round(size / self.lot_size) * self.lot_size

        return size


class Episode:
    """Container for a single trading episode trajectory."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def add_step(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            log_prob: torch.Tensor,
            value: torch.Tensor,
            done: bool,
    ):
        """Add a single timestep to the episode."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, gamma: float = 0.99) -> List[float]:
        """Compute discounted returns for each timestep."""
        returns = []
        G = 0.0
        for reward in reversed(self.rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        return returns

    def compute_advantages(
            self, gamma: float = 0.99, lambda_: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        values = torch.stack(self.values)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        advantages = []
        advantage = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantage = delta + gamma * lambda_ * advantage * (1 - dones[t])
            advantages.insert(0, advantage.item())
            next_value = values[t]

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns


def collect_episode(
        signal_model: SignalModel,
        policy: ExecutionPolicy,
        env: SimulatedRetailExecutionEnv,
        price_data: np.ndarray,
        device: torch.device,
        action_converter: ActionConverter,
) -> Episode:
    """
    Collect a single episode trajectory by running the agent in the environment.

    Args:
        signal_model: Model that processes price features
        policy: Policy network that outputs actions
        env: Trading execution environment
        price_data: Price features for the episode (T, F) array
        device: Torch device
        action_converter: Converts policy outputs to OrderActions

    Returns:
        Episode object containing states, actions, rewards, etc.
    """
    episode = Episode()
    obs = env.reset()

    for t in range(len(price_data)):
        # Get current price features
        state = price_data[t]

        # Run through model (add batch and sequence dims)
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            signal_out = signal_model(x)
            embedding = signal_out["embedding"]

        # Get policy action
        policy_logits, value = policy(embedding, detach_signal=False)
        action_probs = F.softmax(policy_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        # Convert to environment action (Phase 3: pass cash for position sizing)
        order_action = action_converter.policy_to_order(
            action.item(), obs["mid_price"], obs["inventory"], obs["cash"]
        )

        # Step environment
        next_obs, reward, done, info = env.step(order_action)

        # Store transition
        episode.add_step(
            state=state,
            action=action.item(),
            reward=reward,
            log_prob=log_prob,
            value=value.squeeze(),
            done=done,
        )

        obs = next_obs

        if done:
            break

    return episode


def update_policy(
        policy: ExecutionPolicy,
        optimizer: torch.optim.Optimizer,
        episode: Episode,
        cfg: RLTrainingConfig,
) -> Dict[str, float]:
    """
    Update policy using collected episode trajectory with A2C-style loss.

    Args:
        policy: Policy network to update
        optimizer: Optimizer for policy parameters
        episode: Collected episode trajectory
        cfg: RL training configuration

    Returns:
        Dictionary of loss metrics
    """
    # Compute advantages and returns
    advantages, returns = episode.compute_advantages(
        gamma=cfg.gamma, lambda_=cfg.gae_lambda
    )

    # Stack tensors
    log_probs = torch.stack(episode.log_probs)
    values = torch.stack(episode.values)

    # Policy gradient loss (negative because we want to maximize)
    policy_loss = -(log_probs * advantages.detach()).mean()

    # Value function loss
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus for exploration
    # Recompute action probs to get entropy (log_probs alone isn't enough)
    # For now, use simple approach - just policy and value loss
    # TODO: Add proper entropy calculation

    total_loss = policy_loss + cfg.value_coef * value_loss

    optimizer.zero_grad()
    total_loss.backward()

    if cfg.grad_clip:
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)

    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "total_loss": total_loss.item(),
        "mean_return": returns.mean().item(),
        "mean_advantage": advantages.mean().item(),
    }


def train_with_environment(
        signal_model: SignalModel,
        policy: ExecutionPolicy,
        train_data: np.ndarray,
        cfg: RLTrainingConfig,
        env_config: ExecutionConfig,
        device: torch.device,
) -> ExecutionPolicy:
    """
    Train policy using real trading environment and PnL rewards.

    Args:
        signal_model: Pre-trained signal model (frozen during training)
        policy: Execution policy to train
        train_data: Training price features (N, T, F) array
        cfg: RL training configuration
        env_config: Execution environment configuration
        device: Torch device

    Returns:
        Trained policy
    """
    # Freeze signal model - set to inference mode
    for param in signal_model.parameters():
        param.requires_grad = False
    signal_model.train(False)

    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    # Phase 3: Initialize ActionConverter with position sizing parameters
    action_converter = ActionConverter(
        lot_size=env_config.lot_size,
        max_position=10.0,  # Maximum inventory (long or short)
        risk_per_trade=0.02,  # Risk 2% of portfolio per trade
        use_dynamic_sizing=True,  # Enable dynamic position sizing
    )

    logger.info(f"Starting environment-based RL training for {cfg.epochs} epochs")
    logger.info(f"Training data shape: {train_data.shape}")

    for epoch in range(1, cfg.epochs + 1):
        epoch_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "total_loss": 0.0,
            "mean_return": 0.0,
        }
        num_episodes = 0

        # Collect episodes from training data
        for episode_idx in range(len(train_data)):
            # Create fresh environment for each episode
            env = SimulatedRetailExecutionEnv(env_config)

            # Get episode price data
            price_sequence = train_data[episode_idx]  # (T, F)

            # Collect episode trajectory
            episode = collect_episode(
                signal_model=signal_model,
                policy=policy,
                env=env,
                price_data=price_sequence,
                device=device,
                action_converter=action_converter,
            )

            # Update policy
            metrics = update_policy(policy, optimizer, episode, cfg)

            # Accumulate metrics
            for key in epoch_metrics:
                epoch_metrics[key] += metrics.get(key, 0.0)
            num_episodes += 1

            if (episode_idx + 1) % 10 == 0:
                logger.info(
                    f"[epoch {epoch}] episode {episode_idx + 1}/{len(train_data)} | "
                    f"return={metrics['mean_return']:.2f} | "
                    f"policy_loss={metrics['policy_loss']:.4f}"
                )

        # Log epoch summary
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_episodes, 1)

        logger.info(
            f"Epoch {epoch}/{cfg.epochs} complete | "
            f"avg_return={epoch_metrics['mean_return']:.2f} | "
            f"policy_loss={epoch_metrics['policy_loss']:.4f} | "
            f"value_loss={epoch_metrics['value_loss']:.4f}"
        )

    logger.info("Environment-based RL training complete")
    return policy


def main():
    parser = argparse.ArgumentParser(
        description="Train RL policy using execution environment with real PnL rewards"
    )
    parser.add_argument("--signal-model-path", required=True, help="Path to pre-trained signal model")
    parser.add_argument("--train-data-path", required=True, help="Path to training data (.npy)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--grad-clip", type=float, default=0.5, help="Gradient clipping")
    parser.add_argument("--checkpoint-path", default="models/env_policy.pt", help="Save path")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    # Load signal model (frozen)
    # TODO: Add proper model loading from checkpoint
    logger.info(f"Loading signal model from {args.signal_model_path}")
    # signal_model = torch.load(args.signal_model_path, map_location=device)

    # Load training data
    logger.info(f"Loading training data from {args.train_data_path}")
    # train_data = np.load(args.train_data_path)

    # For now, create placeholder
    logger.warning("Using placeholder model and data - TODO: implement proper loading")
    signal_cfg = SignalModelConfig(num_features=50)
    signal_model = SignalModel(signal_cfg).to(device)

    train_data = np.random.randn(100, 390, 50).astype(np.float32)  # (N, T, F)

    # Create policy
    policy_cfg = PolicyConfig(
        input_dim=signal_model.signal_dim,
        hidden_dim=128,
        num_actions=3,  # SELL, HOLD, BUY
    )
    policy = ExecutionPolicy(policy_cfg).to(device)

    # RL training config
    rl_cfg = RLTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        value_coef=args.value_coef,
        grad_clip=args.grad_clip,
    )

    # Environment config
    env_cfg = ExecutionConfig(
        initial_cash=50_000.0,
        lot_size=1.0,
        spread=0.02,
        time_horizon=390,  # 6.5 hour trading day in minutes
    )

    # Train policy
    trained_policy = train_with_environment(
        signal_model=signal_model,
        policy=policy,
        train_data=train_data,
        cfg=rl_cfg,
        env_config=env_cfg,
        device=device,
    )

    # Save checkpoint
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_policy.state_dict(), checkpoint_path)
    logger.info(f"Saved policy checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
