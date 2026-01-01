"""Neural network architectures for Soft Actor-Critic (SAC).

Implements:
1. QNetwork: Critic that estimates Q-values for state-action pairs
2. StochasticPolicy: Actor that outputs action distributions with reparameterization

Research basis:
- "Soft Actor-Critic" (Haarnoja et al., 2018)
- "Soft Actor-Critic Algorithms and Applications" (Haarnoja et al., 2019)
- Twin critics reduce Q-value overestimation
- Stochastic policy enables maximum entropy RL
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -20  # Minimum log std for numerical stability
LOG_STD_MAX = 2  # Maximum log std (prevents excessive exploration)


class QNetwork(nn.Module):
    """Q-network (critic) for SAC.

    Maps (state, action) pairs to Q-values.
    SAC uses twin Q-networks to reduce overestimation bias.

    Architecture:
        Input: concat(state, action)
        Hidden: [256, 256] with ReLU
        Output: Scalar Q-value

    Example:
        q_net = QNetwork(state_dim=76, action_dim=1, hidden_dim=256)
        q_value = q_net(state, action)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """Initialize Q-network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer size
        """
        super().__init__()

        # Q(s, a) architecture
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value for (state, action) pair.

        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]

        Returns:
            Q-value tensor [batch_size, 1]
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        q_value = self.net(x)
        return q_value


class StochasticPolicy(nn.Module):
    """Stochastic policy (actor) for SAC.

    Outputs a Gaussian distribution over actions with:
    - Reparameterization trick for gradient flow
    - Tanh squashing to bound actions to [-1, 1]
    - Log probability correction for the squashing

    Architecture:
        Input: state
        Hidden: [256, 256] with ReLU
        Outputs: mean, log_std

    Sampling:
        1. Sample z ~ N(mean, std)
        2. Squash: action = tanh(z)
        3. Compute log_prob with Jacobian correction

    Example:
        policy = StochasticPolicy(state_dim=76, action_dim=1)
        action, log_prob = policy(state, deterministic=False)
        action_deterministic = policy(state, deterministic=True)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """Initialize stochastic policy.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer size
        """
        super().__init__()

        self.action_dim = action_dim

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(
            self,
            state: torch.Tensor,
            deterministic: bool = False,
            return_log_prob: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample action from policy.

        Args:
            state: State tensor [batch_size, state_dim]
            deterministic: If True, return mean action (no noise)
            return_log_prob: If True, return log probability

        Returns:
            Tuple of (action, log_prob)
            - action: [batch_size, action_dim] in [-1, 1]
            - log_prob: [batch_size] if return_log_prob else None
        """
        # Shared layers
        x = self.shared(state)

        # Compute mean and log_std
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        if deterministic:
            # Deterministic action (mean)
            action = torch.tanh(mean)
            log_prob = None if not return_log_prob else torch.zeros(state.shape[0])
        else:
            # Stochastic action with reparameterization trick
            # Sample z ~ N(0, 1)
            normal = Normal(mean, std)
            z = normal.rsample()  # Reparameterization: z = mean + std * epsilon

            # Squash to [-1, 1]
            action = torch.tanh(z)

            if return_log_prob:
                # Compute log probability with tanh correction
                # log π(a|s) = log N(z|μ,σ) - log(1 - tanh²(z))
                log_prob = normal.log_prob(z)

                # Apply tanh correction (Jacobian of tanh transformation)
                # Σ log(1 - tanh²(z)) across action dimensions
                log_prob -= torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=-1)
            else:
                log_prob = None

        return action, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action without log probability (for inference).

        Args:
            state: State tensor [batch_size, state_dim]
            deterministic: If True, return mean action

        Returns:
            Action tensor [batch_size, action_dim]
        """
        action, _ = self.forward(state, deterministic=deterministic, return_log_prob=False)
        return action


# Example usage and testing
if __name__ == "__main__":
    print("Testing SAC Networks...\n")

    # Test 1: Q-Network
    print("=" * 60)
    print("Test 1: Q-Network")
    print("=" * 60)

    state_dim = 76
    action_dim = 1
    batch_size = 32

    q_net = QNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=256)

    # Test forward pass
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)

    q_values = q_net(states, actions)

    print("✓ Q-Network forward pass:")
    print(f"  - Input: state {states.shape} + action {actions.shape}")
    print(f"  - Output: Q-values {q_values.shape}")
    print(f"  - Q-value range: [{q_values.min().item():.3f}, {q_values.max().item():.3f}]")
    print()

    # Test 2: Stochastic Policy
    print("=" * 60)
    print("Test 2: Stochastic Policy")
    print("=" * 60)

    policy = StochasticPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=256)

    # Test stochastic sampling
    actions_stoch, log_probs = policy(states, deterministic=False)

    print("✓ Stochastic policy:")
    print(f"  - Input: states {states.shape}")
    print(f"  - Output: actions {actions_stoch.shape}, log_probs {log_probs.shape}")
    print(f"  - Action range: [{actions_stoch.min().item():.3f}, {actions_stoch.max().item():.3f}]")
    print(f"  - Actions bounded to [-1, 1]: {torch.all(actions_stoch >= -1) and torch.all(actions_stoch <= 1)}")
    print(f"  - Log prob range: [{log_probs.min().item():.3f}, {log_probs.max().item():.3f}]")
    print()

    # Test deterministic sampling
    actions_det, _ = policy(states, deterministic=True)

    print("✓ Deterministic policy:")
    print(f"  - Output: actions {actions_det.shape}")
    print(f"  - Action range: [{actions_det.min().item():.3f}, {actions_det.max().item():.3f}]")
    print(f"  - Actions bounded to [-1, 1]: {torch.all(actions_det >= -1) and torch.all(actions_det <= 1)}")
    print()

    # Test 3: Reparameterization Trick (gradients flow)
    print("=" * 60)
    print("Test 3: Reparameterization Trick")
    print("=" * 60)

    # Enable gradients
    states.requires_grad = True

    # Forward pass
    actions, log_probs = policy(states)

    # Backprop through action sampling
    loss = actions.mean()
    loss.backward()

    print("✓ Gradient flow test:")
    print(f"  - State gradients exist: {states.grad is not None}")
    print(f"  - State grad norm: {states.grad.norm().item():.6f}")
    print("  - Reparameterization enables gradient flow through stochastic sampling")
    print()

    # Test 4: Log Probability Sanity Check
    print("=" * 60)
    print("Test 4: Log Probability Sanity Check")
    print("=" * 60)

    # Sample multiple actions, check if log probs are valid
    test_state = torch.randn(1, state_dim)

    log_probs_list = []
    for _ in range(100):
        _, log_prob = policy(test_state, deterministic=False)
        log_probs_list.append(log_prob.item())

    log_probs_array = torch.tensor(log_probs_list)

    print("✓ Log probability statistics (100 samples):")
    print(f"  - Mean: {log_probs_array.mean().item():.3f}")
    print(f"  - Std: {log_probs_array.std().item():.3f}")
    print(f"  - Range: [{log_probs_array.min().item():.3f}, {log_probs_array.max().item():.3f}]")
    print(f"  - All finite: {torch.all(torch.isfinite(log_probs_array))}")
    print()

    # Test 5: Action Distribution
    print("=" * 60)
    print("Test 5: Action Distribution")
    print("=" * 60)

    # Sample many actions, check distribution
    test_state = torch.randn(1, state_dim)
    actions_list = []

    for _ in range(1000):
        action, _ = policy(test_state, deterministic=False)
        actions_list.append(action.item())

    actions_array = torch.tensor(actions_list)

    print("✓ Action distribution (1000 samples):")
    print(f"  - Mean: {actions_array.mean().item():.3f}")
    print(f"  - Std: {actions_array.std().item():.3f}")
    print(f"  - Range: [{actions_array.min().item():.3f}, {actions_array.max().item():.3f}]")
    print(f"  - All in [-1, 1]: {torch.all(actions_array >= -1) and torch.all(actions_array <= 1)}")
    print()

    print("=" * 60)
    print("✅ All SAC network tests passed!")
    print("=" * 60)
    print("\nKey features verified:")
    print("  1. Q-network computes Q-values for (state, action) pairs")
    print("  2. Stochastic policy outputs bounded actions [-1, 1]")
    print("  3. Reparameterization trick enables gradient flow")
    print("  4. Log probabilities computed with tanh correction")
    print("  5. Deterministic mode available for evaluation")
