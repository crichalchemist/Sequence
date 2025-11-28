from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.agent_hybrid import HybridCNNLSTMAttention


def _collect_outputs(
    model: HybridCNNLSTMAttention, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if isinstance(y, dict):
                y_primary = y["primary"].to(device)
            else:
                y_primary = y.to(device)
            outputs, _ = model(x)
            logits = outputs["primary"]
            preds.append(logits.detach().cpu().numpy())
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
    model: HybridCNNLSTMAttention, loader: DataLoader, task_type: str = "classification"
) -> Dict[str, float]:
    device = next(model.parameters()).device
    logits_or_preds, targets = _collect_outputs(model, loader, device)
    if task_type == "classification":
        return classification_metrics(logits_or_preds, targets)
    return regression_metrics(logits_or_preds, targets)
