from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from config.config import TrainingConfig
from models.agent_hybrid import HybridCNNLSTMAttention


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


def _select_outputs(outputs: Dict[str, torch.Tensor], task_type: str) -> torch.Tensor:
    if task_type == "classification":
        return outputs["direction_logits"]
    return outputs["return"]


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
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    metric_accum = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            x, y = _to_device(batch, device)
            outputs, _ = model(x)
            logits = _select_outputs(outputs, task_type)
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
) -> Dict[str, List[float]]:
    device_str = cfg.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    model.to(device)

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
            outputs, _ = model(x)
            logits = _select_outputs(outputs, task_type)
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
        val_loss, val_metric = _evaluate(model, val_loader, loss_fn, device, task_type)

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
