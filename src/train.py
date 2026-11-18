"""
Model Training and Transfer Learning Pipeline Module
Implements two-stage fine-tuning (head warmup + selective backbone fine-tuning),
Smooth BCE loss with label smoothing, Cosine Annealing, and checkpoint preservation.
"""

import os
import sys
import json
import random
import argparse

# Ensure project root is in sys.path for direct CLI execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.model import (
    build_model,
    freeze_backbone,
    unfreeze_top_stages,
    unfreeze_all_backbone,
    get_parameter_groups,
    save_checkpoint,
    ForensicClassifier
)
from src.dataset import create_dataloader


def set_seed(seed: int = 42) -> None:
    """Pins deterministic random seeds across Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SmoothBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross-Entropy with Logits incorporating label smoothing to prevent overconfidence
    and positive class weighting to counteract False Negatives on subtle deepfakes.
    """
    def __init__(self, epsilon: float = 0.05, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # y_smooth = y * (1 - eps) + 0.5 * eps
        smoothed_targets = targets * (1.0 - self.epsilon) + 0.5 * self.epsilon
        return self.bce(logits, smoothed_targets)


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Tuple[float, float]:
    """Executes a single training epoch and returns mean loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets, _ in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """Evaluates validation partition and computes loss, accuracy, precision, recall, and Macro-F1."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            total_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            y_true = targets.cpu().numpy().flatten()

            all_probs.extend(probs)
            all_targets.extend(y_true)

    n_samples = max(1, len(all_targets))
    avg_loss = total_loss / n_samples
    y_true_arr = np.array(all_targets)
    y_pred_arr = (np.array(all_probs) >= 0.5).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 0 else 0.0
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    return {
        "val_loss": avg_loss,
        "val_accuracy": acc,
        "val_precision": prec,
        "val_recall": rec,
        "val_macro_f1": macro_f1
    }


def train_model(
    train_manifest: List[Dict[str, Any]],
    val_manifest: List[Dict[str, Any]],
    train_config: Dict[str, Any],
    preprocess_config: Dict[str, Any],
    output_checkpoint_path: str = "models/best_model.pth",
    history_output_path: str = "reports/training_history.json"
) -> Dict[str, Any]:
    """
    Executes the complete two-stage transfer learning pipeline.
    """
    seed = train_config.get("seed", 42)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on compute device: {device.upper()}")

    architecture = preprocess_config.get("model_architecture", "efficientnet_b0")
    batch_size = train_config.get("batch_size", 32)
    num_workers = train_config.get("num_workers", 0)

    train_loader = create_dataloader(train_manifest, batch_size=batch_size, is_training=True, num_workers=num_workers)
    val_loader = create_dataloader(val_manifest, batch_size=batch_size, is_training=False, num_workers=num_workers)

    model = build_model(architecture=architecture, pretrained=True)
    model.to(device)

    label_smoothing = train_config.get("label_smoothing", 0.05)
    pos_weight_val = train_config.get("pos_weight", 1.4)
    pos_weight = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)
    criterion = SmoothBCEWithLogitsLoss(epsilon=label_smoothing, pos_weight=pos_weight)

    history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = 0

    # --------------------------------------------------------------------------
    # STAGE 1: Head Warmup (Backbone Frozen)
    # --------------------------------------------------------------------------
    stage1_epochs = train_config.get("stage1_epochs", 2)
    stage1_lr = train_config.get("stage1_lr", 1e-3)
    freeze_backbone(model)

    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    optimizer = AdamW(head_params, lr=stage1_lr, weight_decay=train_config.get("weight_decay", 1e-4))

    print("\n--- Starting Stage 1: Classifier Head Warmup ---")
    for epoch in range(1, stage1_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "stage": 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            **val_metrics
        }
        history.append(epoch_record)
        print(f"Epoch {epoch:02d} [Stage 1] - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val F1: {val_metrics['val_macro_f1']:.4f} Loss: {val_metrics['val_loss']:.4f}")

        if val_metrics["val_macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["val_macro_f1"]
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, val_metrics, output_checkpoint_path)

    # --------------------------------------------------------------------------
    # STAGE 2: Full Backbone End-to-End Fine-Tuning
    # --------------------------------------------------------------------------
    stage2_epochs = train_config.get("stage2_epochs", 6)
    stage2_backbone_lr = train_config.get("stage2_backbone_lr", 3e-5)
    stage2_head_lr = train_config.get("stage2_head_lr", 5e-4)
    patience = train_config.get("patience", 3)

    unfreeze_all_backbone(model)
    param_groups = get_parameter_groups(
        model,
        backbone_lr=stage2_backbone_lr,
        head_lr=stage2_head_lr,
        weight_decay=train_config.get("weight_decay", 1e-4)
    )
    optimizer = AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs, eta_min=1e-6)

    print("\n--- Starting Stage 2: Full Backbone End-to-End Fine-Tuning ---")
    patience_counter = 0

    for epoch_idx in range(1, stage2_epochs + 1):
        current_epoch = stage1_epochs + epoch_idx
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": current_epoch,
            "stage": 2,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            **val_metrics
        }
        history.append(epoch_record)
        print(f"Epoch {current_epoch:02d} [Stage 2] - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val F1: {val_metrics['val_macro_f1']:.4f} Loss: {val_metrics['val_loss']:.4f}")

        if val_metrics["val_macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["val_macro_f1"]
            best_epoch = current_epoch
            patience_counter = 0
            save_checkpoint(model, optimizer, current_epoch, val_metrics, output_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {current_epoch}. Best Val F1 was {best_val_f1:.4f} at epoch {best_epoch}.")
                break

    os.makedirs(os.path.dirname(os.path.abspath(history_output_path)), exist_ok=True)
    with open(history_output_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining Complete. Best checkpoint saved to '{output_checkpoint_path}' (Val Macro-F1: {best_val_f1:.4f})")
    return {"best_epoch": best_epoch, "best_val_macro_f1": best_val_f1, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training CLI script.")
    parser.add_argument("--config", type=str, default="configs/train_config.json")
    parser.add_argument("--preprocess", type=str, default="configs/preprocess.json")
    parser.add_argument("--train_manifest", type=str, default="manifests/train_manifest.json")
    parser.add_argument("--val_manifest", type=str, default="manifests/val_manifest.json")
    parser.add_argument("--output_model", type=str, default="models/best_model.pth")
    parser.add_argument("--history", type=str, default="reports/training_history.json")
    args = parser.parse_args()

    print(f"Loading configurations...")
    with open(args.config, "r") as f:
        train_config = json.load(f)
    with open(args.preprocess, "r") as f:
        preprocess_config = json.load(f)
    with open(args.train_manifest, "r") as f:
        train_manifest = json.load(f)
    with open(args.val_manifest, "r") as f:
        val_manifest = json.load(f)

    print(f"Loaded {len(train_manifest)} training samples and {len(val_manifest)} validation samples.")
    train_model(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        train_config=train_config,
        preprocess_config=preprocess_config,
        output_checkpoint_path=args.output_model,
        history_output_path=args.history
    )
