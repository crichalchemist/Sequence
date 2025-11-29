from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.agent_hybrid import HybridCNNLSTMAttention
from models.signal_policy import SignalPolicyAgent
from risk.risk_manager import RiskManager


def _collect_outputs(
    model: HybridCNNLSTMAttention,
    loader: DataLoader,
    device: torch.device,
    risk_manager: RiskManager | None = None,
    task_type: str = "classification",
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs, _ = model(x)
            primary = outputs["primary"]
            if risk_manager:
                context = risk_manager.build_context(x=x)
                if task_type == "classification":
                    primary, reasons = risk_manager.apply_classification_logits(primary, context)
                else:
                    primary, reasons = risk_manager.apply_regression_output(primary, context)
                risk_manager.log_events(reasons, prefix="eval")
            y_primary = y["primary"].to(device) if isinstance(y, dict) else y.to(device)
            preds.append(primary.detach().cpu().numpy())
            targets.append(y_primary.detach().cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def classification_metrics(logits: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    pred_labels = logits.argmax(axis=1)
    accuracy = (pred_labels == targets).mean()

    num_classes = logits.shape[1]
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, pred_labels):
        confusion[int(t), int(p)] += 1

    precision_list = []
    recall_list = []
    f1_list = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(np.mean(precision_list)),
        "recall_macro": float(np.mean(recall_list)),
        "f1_macro": float(np.mean(f1_list)),
        "confusion_matrix": confusion,
    }


def regression_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    preds_flat = preds.squeeze()
    targets_flat = targets.squeeze()
    mse = np.mean((preds_flat - targets_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_flat - targets_flat))
    target_mean = targets_flat.mean()
    ss_tot = np.sum((targets_flat - target_mean) ** 2)
    ss_res = np.sum((targets_flat - preds_flat) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def evaluate_model(
    model: HybridCNNLSTMAttention,
    loader: DataLoader,
    task_type: str = "classification",
    risk_manager: RiskManager | None = None,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    logits_or_preds, targets = _collect_outputs(
        model, loader, device, risk_manager=risk_manager, task_type=task_type
    )
    if task_type == "classification":
        return classification_metrics(logits_or_preds, targets)
    return regression_metrics(logits_or_preds, targets)


def evaluate_policy_agent(
    agent: SignalPolicyAgent, loader: DataLoader, task_type: str = "classification"
) -> Dict[str, float]:
    device = next(agent.parameters()).device
    agent.eval()
    total = 0
    correct = 0
    value_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = agent(x, detach_signal=True)
            actions = out["policy_logits"].argmax(dim=1)
            if task_type == "classification":
                correct += (actions == y).sum().item()
                total += y.numel()
            else:
                target_actions = (y.squeeze(-1) > 0).long()
                correct += (actions == target_actions).sum().item()
                total += target_actions.numel()
            value_sum += out["value"].detach().cpu().sum().item()

    accuracy = correct / total if total > 0 else 0.0
    return {"action_accuracy": accuracy, "avg_value": value_sum / max(1, total)}
