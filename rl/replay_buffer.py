"""Experience replay buffers for off-policy RL algorithms.

Implements both uniform and prioritized experience replay for sample-efficient learning.

Key Concepts:
- Uniform replay: Sample transitions uniformly at random
- Prioritized replay: Sample proportional to TD error (important transitions more often)
- Importance sampling: Correct bias introduced by prioritized sampling

Research basis:
- Uniform replay: "Human-level control through deep RL" (Mnih et al., 2015)
- Prioritized replay: "Prioritized Experience Replay" (Schaul et al., 2016)
- Improves sample efficiency by 2-4x for continuous control tasks
"""

import numpy as np


class ReplayBuffer:
    """Uniform experience replay buffer.

    Stores transitions (s, a, r, s', done) and samples uniformly at random.
    Used by off-policy algorithms like SAC, DDPG, TD3.

    Example:
        buffer = ReplayBuffer(capacity=100_000)

        # Store transitions
        buffer.add(state, action, reward, next_state, done)

        # Sample batch
        batch = buffer.sample(batch_size=256)
        states, actions, rewards, next_states, dones = batch
    """

    def __init__(self, capacity: int):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer: list[tuple] = []
        self.position = 0

    def add(
            self,
            state: np.ndarray,
            action: float,
            reward: float,
            next_state: np.ndarray,
            done: bool,
    ) -> None:
        """Add transition to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        transition = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            # Overwrite oldest transition (circular buffer)
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Sample batch of transitions uniformly at random.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        # Sample random indices
        indices = np.random.randint(0, len(self.buffer), size=batch_size)

        # Gather transitions
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[idx] for idx in indices]
        )

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer.

    Samples transitions proportional to their TD error (priority).
    Uses importance sampling weights to correct bias.

    Algorithm:
        1. Store transitions with initial priority p_i
        2. Sample with probability P(i) = p_i^α / Σ p_j^α
        3. Weight samples by w_i = (N * P(i))^(-β)
        4. Update priorities after training: p_i = |TD_error_i| + ε

    Hyperparameters:
        - α: How much to prioritize (0=uniform, 1=full prioritization)
        - β: Importance sampling correction (0=no correction, 1=full correction)
        - ε: Small constant to ensure non-zero probabilities

    Example:
        buffer = PrioritizedReplayBuffer(
            capacity=100_000,
            alpha=0.6,  # Moderate prioritization
            beta_start=0.4,  # Anneal to 1.0
            beta_frames=100_000
        )

        # Add transition with default priority
        buffer.add(state, action, reward, next_state, done)

        # Sample batch with importance weights
        batch, indices, weights = buffer.sample(batch_size=256)

        # Compute TD errors, update priorities
        td_errors = compute_td_errors(batch)
        buffer.update_priorities(indices, td_errors)
    """

    def __init__(
            self,
            capacity: int,
            alpha: float = 0.6,
            beta_start: float = 0.4,
            beta_frames: int = 100_000,
            epsilon: float = 1e-6,
    ):
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            alpha: Prioritization exponent (0=uniform, 1=full)
            beta_start: Initial importance sampling correction
            beta_frames: Frames over which to anneal beta to 1.0
            epsilon: Small constant for numerical stability
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1  # Current frame count for beta annealing

        # Priority storage (same size as buffer)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0  # Track max priority for new transitions

    def add(
            self,
            state: np.ndarray,
            action: float,
            reward: float,
            next_state: np.ndarray,
            done: bool,
    ) -> None:
        """Add transition with maximum priority.

        New transitions get max priority to ensure they're sampled at least once.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        # Store transition
        super().add(state, action, reward, next_state, done)

        # Assign max priority to new transition
        self.priorities[self.position - 1] = self.max_priority

    def sample(
            self, batch_size: int
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling and importance weights.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of:
                - batch: (states, actions, rewards, next_states, dones)
                - indices: Sampled indices (for priority updates)
                - weights: Importance sampling weights
        """
        # Get current buffer size
        N = len(self.buffer)

        # Compute sampling probabilities
        priorities = self.priorities[:N]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        # Sample indices based on priorities
        indices = np.random.choice(N, size=batch_size, p=probs)

        # Gather transitions
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[idx] for idx in indices]
        )

        # Compute importance sampling weights
        # w_i = (N * P(i))^(-β)
        beta = self._get_beta()
        weights = (N * probs[indices]) ** (-beta)

        # Normalize weights (for stability)
        weights = weights / weights.max()

        batch = (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors.

        Args:
            indices: Indices of sampled transitions
            td_errors: Absolute TD errors for each transition
        """
        # Update priorities: p_i = |TD_error| + ε
        priorities = np.abs(td_errors) + self.epsilon

        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

        # Track max priority for new transitions
        self.max_priority = max(self.max_priority, priorities.max())

        # Increment frame count
        self.frame += 1

    def _get_beta(self) -> float:
        """Get current beta value (anneals from beta_start to 1.0).

        Returns:
            Current beta value
        """
        # Linear annealing
        progress = min(self.frame / self.beta_frames, 1.0)
        return self.beta_start + progress * (1.0 - self.beta_start)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Replay Buffers...\n")

    # Test 1: Uniform Replay Buffer
    print("=" * 60)
    print("Test 1: Uniform Replay Buffer")
    print("=" * 60)

    buffer = ReplayBuffer(capacity=1000)

    # Add transitions
    for i in range(100):
        state = np.random.randn(76)
        action = np.random.uniform(-1, 1)
        reward = np.random.randn()
        next_state = np.random.randn(76)
        done = i % 20 == 0  # Episode ends every 20 steps

        buffer.add(state, action, reward, next_state, done)

    print(f"Buffer size: {len(buffer)} / {buffer.capacity}")

    # Sample batch
    batch = buffer.sample(batch_size=32)
    states, actions, rewards, next_states, dones = batch

    print("✓ Sampled batch:")
    print(f"  - States shape: {states.shape}")
    print(f"  - Actions shape: {actions.shape}")
    print(f"  - Rewards shape: {rewards.shape}")
    print(f"  - Next states shape: {next_states.shape}")
    print(f"  - Dones shape: {dones.shape}")
    print()

    # Test 2: Prioritized Replay Buffer
    print("=" * 60)
    print("Test 2: Prioritized Replay Buffer")
    print("=" * 60)

    pri_buffer = PrioritizedReplayBuffer(
        capacity=1000, alpha=0.6, beta_start=0.4, beta_frames=10_000
    )

    # Add transitions
    for i in range(100):
        state = np.random.randn(76)
        action = np.random.uniform(-1, 1)
        reward = np.random.randn()
        next_state = np.random.randn(76)
        done = i % 20 == 0

        pri_buffer.add(state, action, reward, next_state, done)

    print(f"Buffer size: {len(pri_buffer)} / {pri_buffer.capacity}")

    # Sample batch with priorities
    batch, indices, weights = pri_buffer.sample(batch_size=32)
    states, actions, rewards, next_states, dones = batch

    print("✓ Sampled prioritized batch:")
    print(f"  - States shape: {states.shape}")
    print(f"  - Indices shape: {indices.shape}")
    print(f"  - Weights shape: {weights.shape}")
    print(f"  - Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print()

    # Simulate priority updates
    td_errors = np.abs(np.random.randn(32))  # Simulated TD errors
    pri_buffer.update_priorities(indices, td_errors)

    print("✓ Updated priorities:")
    print(f"  - Max priority: {pri_buffer.max_priority:.3f}")
    print(f"  - Current beta: {pri_buffer._get_beta():.3f}")
    print()

    # Test 3: Priority Distribution
    print("=" * 60)
    print("Test 3: Priority Distribution Check")
    print("=" * 60)

    # Create buffer with known priorities
    test_buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)

    # Add 10 transitions
    for i in range(10):
        test_buffer.add(
            np.random.randn(76), 0.0, 0.0, np.random.randn(76), False
        )

    # Set specific priorities (high priority for index 0)
    test_priorities = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    test_buffer.priorities[: len(test_priorities)] = test_priorities

    # Sample many times, check distribution
    sample_counts = np.zeros(10)
    n_samples = 1000

    for _ in range(n_samples):
        _, indices, _ = test_buffer.sample(batch_size=1)
        sample_counts[indices[0]] += 1

    print("Sample distribution (index 0 has 10x priority):")
    print(f"  Index 0: {sample_counts[0]} samples ({sample_counts[0] / n_samples:.1%})")
    print(f"  Index 1-9: {sample_counts[1:].mean():.0f} samples avg ({sample_counts[1:].mean() / n_samples:.1%})")

    # Index 0 should be sampled much more
    assert (
            sample_counts[0] > sample_counts[1:].mean() * 2
    ), "Priority sampling not working correctly"

    print("✓ Prioritized sampling working correctly\n")

    print("=" * 60)
    print("✅ All replay buffer tests passed!")
    print("=" * 60)
