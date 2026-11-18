"""
Post-Processing Probability Calibration Module
Implements Platt Temperature Scaling optimization via L-BFGS to calibrate
raw neural network logits against empirical validation probabilities.
"""

import os
import sys
import json

# Ensure project root is in sys.path for direct CLI execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar


class TemperatureScaler(nn.Module):
    """
    Applies learned scalar temperature parameter T to rescale logits before sigmoid activation.
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        return logits / temp


def fit_temperature(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor
) -> float:
    """
    Optimizes temperature scaling parameter T on validation split logits by minimizing Negative Log-Likelihood (NLL).
    
    Args:
        val_logits: Tensor of raw model logits with shape [N, 1] or [N].
        val_labels: Ground truth binary labels with shape [N, 1] or [N] in {0.0, 1.0}.
        
    Returns:
        float: Optimal scalar temperature T in range [0.1, 10.0].
    """
    logits_np = val_logits.detach().cpu().numpy().flatten().astype(np.float64)
    labels_np = val_labels.detach().cpu().numpy().flatten().astype(np.float64)

    def nll_objective(temp: float) -> float:
        scaled_logits = logits_np / max(1e-4, temp)
        # Numerically stable binary cross-entropy:
        # log(1 + exp(-x)) for positive, and -x + log(1 + exp(x)) for negative
        log_prob_pos = -np.logaddexp(0.0, -scaled_logits)
        log_prob_neg = -np.logaddexp(0.0, scaled_logits)
        loss = -np.mean(labels_np * log_prob_pos + (1.0 - labels_np) * log_prob_neg)
        return float(loss)

    res = minimize_scalar(nll_objective, bounds=(0.5, 3.0), method="bounded")
    optimal_t = float(res.x)
    optimal_t = max(0.5, min(3.0, optimal_t))
    return round(optimal_t, 4)


def calibrate_score(raw_logit: float, temperature: float = 1.0) -> float:
    """
    Transforms raw scalar logit into calibrated probability score in range [0.0, 1.0].
    
    Args:
        raw_logit: Output logit from neural network.
        temperature: Calibrated temperature parameter T.
        
    Returns:
        float: Calibrated probability in [0.0, 1.0].
    """
    temp = max(0.1, min(10.0, temperature))
    scaled_logit = raw_logit / temp
    # Numerically stable sigmoid
    if scaled_logit >= 0:
        prob = 1.0 / (1.0 + np.exp(-scaled_logit))
    else:
        prob = np.exp(scaled_logit) / (1.0 + np.exp(scaled_logit))
    return float(round(float(prob), 4))


def update_preprocess_config_temperature(
    config_path: str,
    fitted_temperature: float
) -> None:
    """Updates the temperature field in configs/preprocess.json."""
    with open(config_path, "r") as f:
        config = json.load(f)
        
    config.setdefault("calibration", {})["temperature"] = fitted_temperature
    config["calibration"]["method"] = "platt_scaling_validation"
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    import argparse
    from src.model import build_model, load_checkpoint
    from src.dataset import create_dataloader

    parser = argparse.ArgumentParser(description="Temperature scaling CLI.")
    parser.add_argument("--val_manifest", type=str, default="manifests/val_manifest.json")
    parser.add_argument("--weights", type=str, default="models/best_model.pth")
    parser.add_argument("--config", type=str, default="configs/preprocess.json")
    args = parser.parse_args()

    print(f"Running Temperature Calibration on '{args.val_manifest}' using weights '{args.weights}'...")
    with open(args.config, "r") as f:
        config = json.load(f)
    with open(args.val_manifest, "r") as f:
        val_manifest = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    architecture = config.get("model_architecture", "efficientnet_b0")
    model = build_model(architecture=architecture, pretrained=False)
    load_checkpoint(args.weights, model, device=device)
    model.eval()

    val_loader = create_dataloader(val_manifest, batch_size=32, is_training=False, num_workers=0)
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_labels.append(targets.cpu())

    if all_logits:
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        optimal_t = fit_temperature(logits_cat, labels_cat)
        update_preprocess_config_temperature(args.config, optimal_t)
        print(f"Temperature fitted successfully: T = {optimal_t}")
        print(f"Updated '{args.config}' with calibrated temperature.")
    else:
        print("Validation loader was empty. Could not fit temperature.")
