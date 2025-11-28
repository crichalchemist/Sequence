from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.config import PolicyConfig, RLTrainingConfig, TrainingConfig
from models.agent_hybrid import HybridCNNLSTMAttention
from models.signal_policy import ExecutionPolicy, SignalModel


def _to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


def _classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / len(targets)


def _regression_rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds, targets = _align_regression(preds, targets)
    mse = torch.mean((preds - targets) ** 2)
    return torch.sqrt(mse).item()


def _align_regression(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    preds = preds.squeeze(-1)
    targets = targets.squeeze(-1)
    return preds, targets


def _evaluate(
    model: HybridCNNLSTMAttention,
    loader: DataLoader,
    loss_fn,
    device,
    task_type: str,
    risk_manager: RiskManager | None = None,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    metric_accum = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            x, y = _to_device(batch, device)
            logits, _ = model(x)
            context = risk_manager.build_context(x=x) if risk_manager else None
            if risk_manager:
                if task_type == "classification":
                    logits, reasons = risk_manager.apply_classification_logits(logits, context)
                else:
                    logits, reasons = risk_manager.apply_regression_output(logits, context)
                risk_manager.log_events(reasons, prefix="eval")
            if task_type == "regression":
                logits, y = _align_regression(logits, y)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            if task_type == "classification":
                metric_accum += _classification_metrics(logits, y)
            else:
                metric_accum += _regression_rmse(logits, y)
            batches += 1
    return total_loss / max(1, batches), metric_accum / max(1, batches)


def train_model(
    model: HybridCNNLSTMAttention,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainingConfig,
    loss_fn=None,
    scheduler=None,
    task_type: str = "classification",
    risk_manager: RiskManager | None = None,
) -> Dict[str, List[float]]:
    device_str = cfg.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    model.to(device)

    if risk_manager is None and getattr(cfg, "risk", None) and cfg.risk.enabled:
        risk_manager = RiskManager(cfg.risk)

    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss() if task_type == "classification" else torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    history = {"train_loss": [], "val_loss": [], "val_metric": []}
    best_metric = -float("inf") if task_type == "classification" else float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            x, y = _to_device(batch, device)
            logits, _ = model(x)
            context = risk_manager.build_context(x=x) if risk_manager else None
            if risk_manager:
                if task_type == "classification":
                    logits, reasons = risk_manager.apply_classification_logits(logits, context)
                    actions = logits.argmax(dim=1)
                    risk_manager.record_actions(actions)
                else:
                    logits, reasons = risk_manager.apply_regression_output(logits, context)
                risk_manager.log_events(reasons, prefix=f"epoch {epoch} step {step}")
            if task_type == "regression":
                logits, y = _align_regression(logits, y)
            loss = loss_fn(logits, y)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            if step % cfg.log_every == 0:
                print(f"epoch {epoch} step {step} loss {running_loss / step:.4f}")

        if scheduler:
            scheduler.step()

        train_epoch_loss = running_loss / max(1, len(train_loader))
        val_loss, val_metric = _evaluate(
            model, val_loader, loss_fn, device, task_type, risk_manager=risk_manager
        )

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)

        is_better = val_metric > best_metric if task_type == "classification" else val_metric < best_metric
        if is_better:
            best_metric = val_metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if cfg.checkpoint_path:
                torch.save(best_state, cfg.checkpoint_path)

        print(
            f"epoch {epoch}/{cfg.epochs} train_loss {train_epoch_loss:.4f} "
            f"val_loss {val_loss:.4f} val_metric {val_metric:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def _signal_losses(signal_out: Dict, targets: torch.Tensor, task_type: str):
    losses = []
    metrics = {}
    aux = signal_out.get("aux", {})
    if "direction_logits" in aux and task_type == "classification":
        cls_loss = torch.nn.functional.cross_entropy(aux["direction_logits"], targets.long())
        losses.append(cls_loss)
        metrics["direction_acc"] = _classification_metrics(aux["direction_logits"], targets)
    if "forecast" in aux:
        forecast_target = targets.float().unsqueeze(-1) if targets.dim() == 1 else targets.float()
        reg_loss = torch.nn.functional.mse_loss(aux["forecast"], forecast_target)
        losses.append(reg_loss)
        metrics["forecast_rmse"] = _regression_rmse(aux["forecast"], forecast_target)
    total_loss = sum(losses) if losses else torch.tensor(0.0, device=targets.device)
    return total_loss, metrics


def pretrain_signal_model(
    signal_model: SignalModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainingConfig,
    task_type: str = "classification",
) -> Dict[str, List[float]]:
    device_str = cfg.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    signal_model.to(device)

    optimizer = torch.optim.Adam(
        signal_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    history = {"train_loss": [], "val_loss": [], "val_metric": []}
    best_metric = -float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        signal_model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            x, y = _to_device(batch, device)
            signal_out = signal_model(x)
            loss, _ = _signal_losses(signal_out, y, task_type)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(signal_model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            if step % cfg.log_every == 0:
                print(f"[signal] epoch {epoch} step {step} loss {running_loss / step:.4f}")

        val_loss, val_metric = _evaluate_signal(signal_model, val_loader, device, task_type)
        history["train_loss"].append(running_loss / max(1, len(train_loader)))
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {k: v.cpu() for k, v in signal_model.state_dict().items()}
            if cfg.checkpoint_path:
                torch.save(best_state, cfg.checkpoint_path)

        print(
            f"[signal] epoch {epoch}/{cfg.epochs} train_loss {history['train_loss'][-1]:.4f} "
            f"val_loss {val_loss:.4f} val_metric {val_metric:.4f}"
        )

    if best_state is not None:
        signal_model.load_state_dict(best_state)

    return history


def _evaluate_signal(signal_model: SignalModel, loader: DataLoader, device, task_type: str):
    signal_model.eval()
    total_loss = 0.0
    total_metric = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            x, y = _to_device(batch, device)
            signal_out = signal_model(x)
            loss, metrics = _signal_losses(signal_out, y, task_type)
            total_loss += loss.item()
            metric_val = metrics.get("direction_acc", metrics.get("forecast_rmse", 0.0))
            total_metric += metric_val
            batches += 1
    return total_loss / max(1, batches), total_metric / max(1, batches)


def _prepare_actions_and_rewards(targets: torch.Tensor, task_type: str):
    if task_type == "classification":
        actions = targets.long()
        reward_lookup = torch.tensor([-1.0, 0.0, 1.0], device=targets.device)
        rewards = reward_lookup[actions.clamp(max=len(reward_lookup) - 1)]
    else:
        rewards = targets.squeeze(-1).float()
        actions = (rewards > 0).long()
    return actions, rewards


def train_execution_policy(
    signal_model: SignalModel,
    policy_head: ExecutionPolicy,
    train_loader: DataLoader,
    cfg: RLTrainingConfig,
    task_type: str = "classification",
):
    device = next(signal_model.parameters()).device
    policy_head.to(device)
    optimizer = torch.optim.Adam(policy_head.parameters(), lr=cfg.learning_rate)

    for epoch in range(1, cfg.epochs + 1):
        policy_head.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            x, y = _to_device(batch, device)
            with torch.no_grad():
                signal_out = signal_model(x)
                signal_embedding = signal_out["embedding"].detach() if cfg.detach_signal else signal_out["embedding"]

            logits, value = policy_head(signal_embedding, detach_signal=False)
            actions, rewards = _prepare_actions_and_rewards(y, task_type)

            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            advantage = rewards - value.detach()

            policy_loss = -(advantage * action_log_probs).mean()
            value_loss = advantage.pow(2).mean()
            entropy = -(probs * log_probs).sum(dim=1).mean()

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(policy_head.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            if step % 50 == 0:
                print(
                    f"[policy] epoch {epoch} step {step} loss {running_loss / step:.4f} "
                    f"policy_loss {policy_loss.item():.4f} value_loss {value_loss.item():.4f}"
                )

        print(f"[policy] epoch {epoch}/{cfg.epochs} loss {running_loss / max(1, len(train_loader)):.4f}")

    if cfg.checkpoint_path:
        torch.save({k: v.cpu() for k, v in policy_head.state_dict().items()}, cfg.checkpoint_path)

    return policy_head
