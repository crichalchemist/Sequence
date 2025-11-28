from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from config.config import MultiTaskLossWeights, TrainingConfig
from models.agent_multitask import MultiHeadHybrid


def _to_device(batch, device):
    x, targets = batch
    x = x.to(device)
    targets = {k: v.to(device) for k, v in targets.items()}
    return x, targets


def _compute_losses(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    loss_weights: MultiTaskLossWeights,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses = {}
    ce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    losses["direction"] = ce(outputs["direction_logits"], targets["direction_class"])
    losses["volatility"] = ce(outputs["volatility_logits"], targets["vol_class"])
    losses["trend"] = ce(outputs["trend_logits"], targets["trend_class"])
    losses["vol_regime"] = ce(outputs["vol_regime_logits"], targets["vol_regime_class"])
    losses["candle_pattern"] = ce(outputs["candle_pattern_logits"], targets["candle_class"])
    losses["return"] = mse(outputs["return"].squeeze(-1), targets["return_reg"])
    losses["next_close"] = mse(outputs["next_close"].squeeze(-1), targets["next_close_reg"])

    total = (
        loss_weights.direction_cls * losses["direction"]
        + loss_weights.vol_cls * losses["volatility"]
        + loss_weights.trend_cls * losses["trend"]
        + loss_weights.vol_regime_cls * losses["vol_regime"]
        + loss_weights.candle_pattern_cls * losses["candle_pattern"]
        + loss_weights.return_reg * losses["return"]
        + loss_weights.next_close_reg * losses["next_close"]
    )
    return total, {k: v.item() for k, v in losses.items()}


def _classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def _rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.squeeze(-1)
    targets = targets.squeeze(-1)
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()


def _evaluate(
    model: MultiHeadHybrid,
    loader: DataLoader,
    loss_weights: MultiTaskLossWeights,
    device,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "dir_acc": 0.0,
        "vol_acc": 0.0,
        "trend_acc": 0.0,
        "vol_regime_acc": 0.0,
        "candle_acc": 0.0,
        "ret_rmse": 0.0,
        "close_rmse": 0.0,
    }
    batches = 0
    with torch.no_grad():
        for batch in loader:
            x, targets = _to_device(batch, device)
            outputs, _ = model(x)
            loss, _ = _compute_losses(outputs, targets, loss_weights)
            totals["loss"] += loss.item()
            totals["dir_acc"] += _classification_accuracy(outputs["direction_logits"], targets["direction_class"])
            totals["vol_acc"] += _classification_accuracy(outputs["volatility_logits"], targets["vol_class"])
            totals["trend_acc"] += _classification_accuracy(outputs["trend_logits"], targets["trend_class"])
            totals["vol_regime_acc"] += _classification_accuracy(
                outputs["vol_regime_logits"], targets["vol_regime_class"]
            )
            totals["candle_acc"] += _classification_accuracy(outputs["candle_pattern_logits"], targets["candle_class"])
            totals["ret_rmse"] += _rmse(outputs["return"], targets["return_reg"])
            totals["close_rmse"] += _rmse(outputs["next_close"], targets["next_close_reg"])
            batches += 1
    if batches == 0:
        return totals
    return {k: v / batches for k, v in totals.items()}


def train_multitask(
    model: MultiHeadHybrid,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainingConfig,
    loss_weights: MultiTaskLossWeights,
) -> Dict[str, List[float]]:
    device_str = cfg.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dir_acc": [],
        "val_vol_acc": [],
        "val_trend_acc": [],
        "val_vol_regime_acc": [],
        "val_candle_acc": [],
    }
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            x, targets = _to_device(batch, device)
            outputs, _ = model(x)
            loss, per_task = _compute_losses(outputs, targets, loss_weights)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            if step % cfg.log_every == 0:
                print(f"epoch {epoch} step {step} loss {running_loss / step:.4f} tasks {per_task}")

        train_epoch_loss = running_loss / max(1, len(train_loader))
        val_metrics = _evaluate(model, val_loader, loss_weights, device)
        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_dir_acc"].append(val_metrics["dir_acc"])
        history["val_vol_acc"].append(val_metrics["vol_acc"])
        history["val_trend_acc"].append(val_metrics["trend_acc"])
        history["val_vol_regime_acc"].append(val_metrics["vol_regime_acc"])
        history["val_candle_acc"].append(val_metrics["candle_acc"])

        is_better = val_metrics["loss"] < best_loss
        if is_better:
            best_loss = val_metrics["loss"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if cfg.checkpoint_path:
                torch.save(best_state, cfg.checkpoint_path)

        print(
            f"epoch {epoch}/{cfg.epochs} train_loss {train_epoch_loss:.4f} "
            f"val_loss {val_metrics['loss']:.4f} "
            f"val_dir_acc {val_metrics['dir_acc']:.4f} val_vol_acc {val_metrics['vol_acc']:.4f} "
            f"val_trend_acc {val_metrics['trend_acc']:.4f} val_vol_regime_acc {val_metrics['vol_regime_acc']:.4f} "
            f"val_candle_acc {val_metrics['candle_acc']:.4f} "
            f"val_ret_rmse {val_metrics['ret_rmse']:.6f} val_close_rmse {val_metrics['close_rmse']:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
