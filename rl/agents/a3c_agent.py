import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import ModelConfig  # noqa: E402
from models.agent_hybrid import TemporalAttention  # noqa: E402


@dataclass
class A3CConfig:
    """Configuration for A3C training and optimization."""

    n_workers: int = 4
    rollout_length: int = 5
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: Optional[float] = 0.5
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    total_steps: int = 100_000
    log_interval: int = 1000
    checkpoint_path: str = "models/a3c_agent.pt"
    device: str = "cpu"


class HybridFeatureExtractor(nn.Module):
    """Shared encoder mirroring the hybrid CNN + LSTM + attention stack."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        padding = cfg.cnn_kernel_size // 2
        self.lstm = nn.LSTM(
            input_size=cfg.num_features,
            hidden_size=cfg.hidden_size_lstm,
            num_layers=cfg.num_layers_lstm,
            batch_first=True,
        )
        self.cnn = nn.Conv1d(
            in_channels=cfg.num_features,
            out_channels=cfg.cnn_num_filters,
            kernel_size=cfg.cnn_kernel_size,
            padding=padding,
        )
        attn_input_dim = cfg.hidden_size_lstm + cfg.cnn_num_filters
        self.attention = TemporalAttention(attn_input_dim, cfg.attention_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.output_dim = attn_input_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return context vector and attention weights."""

        lstm_out, _ = self.lstm(x)
        cnn_in = x.permute(0, 2, 1)
        cnn_features = F.relu(self.cnn(cnn_in)).permute(0, 2, 1)
        combined = torch.cat([lstm_out, cnn_features], dim=-1)
        context, attn_weights = self.attention(combined)
        context = self.dropout(context)
        return context, attn_weights


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network with shared hybrid encoder."""

    def __init__(self, model_cfg: ModelConfig, action_dim: int):
        super().__init__()
        self.encoder = HybridFeatureExtractor(model_cfg)
        self.policy_head = nn.Linear(self.encoder.output_dim, action_dim)
        self.value_head = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, attn_weights = self.encoder(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value, attn_weights

    def act(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value, attn_weights = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy, attn_weights


class SharedAdam(optim.Adam):
    """Adam optimizer with shared states for multiprocessing."""

    def __init__(
        self,
        params,
        lr: float,
        betas: Tuple[float, float],
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay)
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(param.data)
                state["exp_avg_sq"] = torch.zeros_like(param.data)

                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


class A3CAgent:
    """Asynchronous Advantage Actor-Critic training harness."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        a3c_cfg: A3CConfig,
        action_dim: int,
        env_factory: Callable[[], Any],
    ):
        self.model_cfg = model_cfg
        self.cfg = a3c_cfg
        self.env_factory = env_factory
        self.device = torch.device(
            a3c_cfg.device if a3c_cfg.device != "cuda" or torch.cuda.is_available() else "cpu"
        )

        self.global_model = ActorCriticNetwork(model_cfg, action_dim).to(self.device)
        self.global_model.share_memory()
        self.optimizer = SharedAdam(
            self.global_model.parameters(),
            lr=a3c_cfg.learning_rate,
            betas=a3c_cfg.betas,
            weight_decay=a3c_cfg.weight_decay,
        )

        self.global_steps = mp.Value("i", 0)
        self.log_queue: mp.Queue = mp.Queue()

    def save_checkpoint(self) -> None:
        Path(self.cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.global_model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "model_config": self.model_cfg.__dict__,
                "a3c_config": self.cfg.__dict__,
            },
            self.cfg.checkpoint_path,
        )

    def train(self) -> None:
        processes = []
        for worker_id in range(self.cfg.n_workers):
            p = mp.Process(
                target=self.worker_process,
                args=(worker_id,),
                daemon=True,
            )
            p.start()
            processes.append(p)

        last_log = 0
        while any(p.is_alive() for p in processes):
            try:
                msg = self.log_queue.get(timeout=1)
                if msg[0] == "progress":
                    step, loss = msg[1], msg[2]
                    if step - last_log >= self.cfg.log_interval:
                        print(f"[global_step={step}] mean_loss={loss:.4f}")
                        last_log = step
            except Exception:
                pass
            time.sleep(0.1)

        for p in processes:
            p.join()

        self.save_checkpoint()

    def worker_process(self, worker_id: int) -> None:
        env = self.env_factory()
        local_model = ActorCriticNetwork(self.model_cfg, self.global_model.policy_head.out_features).to(
            self.device
        )
        local_model.load_state_dict(self.global_model.state_dict())

        while True:
            with self.global_steps.get_lock():
                if self.global_steps.value >= self.cfg.total_steps:
                    break

            state, _ = self._reset_env(env)
            done = False

            while not done:
                rollout = []
                for _ in range(self.cfg.rollout_length):
                    state_tensor = self._to_tensor(state)
                    action, log_prob, value, entropy, _ = local_model.act(state_tensor)
                    next_state, reward, done, truncated, info = self._step_env(env, action.item())
                    done = bool(done or truncated)

                    rollout.append(
                        {
                            "state": state_tensor,
                            "action": action,
                            "log_prob": log_prob,
                            "value": value,
                            "reward": torch.tensor([reward], device=self.device),
                            "entropy": entropy,
                        }
                    )

                    state = next_state

                    with self.global_steps.get_lock():
                        self.global_steps.value += 1
                        current_step = self.global_steps.value
                    if current_step >= self.cfg.total_steps or done:
                        break

                next_value = torch.tensor(0.0, device=self.device)
                if not done:
                    next_state_tensor = self._to_tensor(state)
                    with torch.no_grad():
                        _, next_value, _ = local_model.forward(next_state_tensor)

                policy_loss, value_loss, entropy_loss = self._compute_losses(
                    rollout, next_value
                )
                total_loss = policy_loss + self.cfg.value_loss_coef * value_loss - self.cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                if self.cfg.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), self.cfg.max_grad_norm)

                for global_param, local_param in zip(self.global_model.parameters(), local_model.parameters()):
                    if global_param.grad is None:
                        global_param.grad = local_param.grad
                    else:
                        global_param.grad.copy_(local_param.grad)

                self.optimizer.step()
                local_model.load_state_dict(self.global_model.state_dict())

                if current_step % self.cfg.log_interval == 0:
                    mean_loss = total_loss.detach().cpu().item()
                    self.log_queue.put(("progress", current_step, mean_loss))

                if current_step >= self.cfg.total_steps or done:
                    break

    def _compute_losses(
        self, rollout: list, next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        returns = []
        R = next_value
        for step in reversed(rollout):
            R = step["reward"] + self.cfg.gamma * R
            returns.insert(0, R)

        log_probs = torch.stack([step["log_prob"] for step in rollout])
        values = torch.stack([step["value"] for step in rollout])
        returns_tensor = torch.stack(returns).squeeze(-1)

        advantages = returns_tensor.detach() - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns_tensor)
        entropy_loss = torch.stack([step["entropy"] for step in rollout]).mean()
        return policy_loss, value_loss, entropy_loss

    def _to_tensor(self, observation: Any) -> torch.Tensor:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _reset_env(env: Any) -> Tuple[Any, Optional[Dict[str, Any]]]:
        result = env.reset()
        if isinstance(result, tuple):
            return result[0], result[1] if len(result) > 1 else None
        return result, None

    @staticmethod
    def _step_env(env: Any, action: int) -> Tuple[Any, float, bool, bool, Optional[Dict[str, Any]]]:
        result = env.step(action)
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            return next_state, float(reward), bool(terminated), bool(truncated), info
        elif len(result) == 4:
            next_state, reward, done, info = result
            return next_state, float(reward), bool(done), False, info
        else:
            raise ValueError("Unexpected environment step return signature")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an A3C agent with the hybrid encoder.")
    parser.add_argument("--env-id", required=True, help="Gym/Gymnasium environment ID (requires the package installed)")
    parser.add_argument("--num-features", type=int, required=True, help="Feature dimension expected by the encoder")
    parser.add_argument("--action-dim", type=int, required=True, help="Number of discrete actions")
    parser.add_argument("--n-workers", type=int, default=4, help="Number of asynchronous workers")
    parser.add_argument("--rollout-length", type=int, default=5, help="Rollout length before each update")
    parser.add_argument("--total-steps", type=int, default=100000, help="Total environment steps across all workers")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="Value loss scaling")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm")
    parser.add_argument("--checkpoint-path", default="models/a3c_agent.pt", help="Where to write checkpoints")
    parser.add_argument("--device", default="cpu", help="Device for computation (cpu or cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import gymnasium as gym
    except ImportError:
        try:
            import gym  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "An environment factory is required. Install gymnasium or provide your own env creator."
            ) from exc
    env_id = args.env_id

    def make_env():
        return gym.make(env_id)

    model_cfg = ModelConfig(num_features=args.num_features)
    a3c_cfg = A3CConfig(
        n_workers=args.n_workers,
        rollout_length=args.rollout_length,
        total_steps=args.total_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        gamma=args.gamma,
        max_grad_norm=args.max_grad_norm,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )

    agent = A3CAgent(
        model_cfg=model_cfg,
        a3c_cfg=a3c_cfg,
        action_dim=args.action_dim,
        env_factory=make_env,
    )
    agent.train()


if __name__ == "__main__":
    main()
