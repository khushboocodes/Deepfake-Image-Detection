"""
Offline Evaluation, Metrics Suite, and Visual Reporting Module
Executes batch evaluation on locked test manifests, calculates complete statistical
and calibration metrics (ROC-AUC, PR-AUC, EER, ECE), renders publication plots,
and extracts top false positives / false negatives.
"""

import os
import sys
import json
import argparse

# Ensure project root is in sys.path for direct CLI execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    brier_score_loss
)

# Force headless Matplotlib backend to prevent GUI display crashes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import build_model, load_checkpoint, ForensicClassifier
from src.dataset import create_dataloader
from src.calibration import calibrate_score


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) across M uniform confidence bins.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_prob)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(round(ece, 4))


def compute_equal_error_rate(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Computes Equal Error Rate (EER) where FPR == FNR (1 - TPR)."""
    fnr = 1.0 - tpr
    diff = np.abs(fpr - fnr)
    idx = np.nanargmin(diff)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(round(float(eer), 4))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Calculates the complete suite of statistical, threshold-independent, and calibration metrics.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp

    acc = float(accuracy_score(y_true, y_pred)) if total > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if total > 0 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr_val = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr_val = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    f1_syn = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_real = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # ROC and PR AUC
    try:
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = float(auc(fpr_arr, tpr_arr))
        eer_val = compute_equal_error_rate(fpr_arr, tpr_arr)
    except Exception:
        roc_auc_val = 0.5
        eer_val = 0.5

    try:
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
        pr_auc_val = float(auc(rec_arr, prec_arr))
    except Exception:
        pr_auc_val = 0.5

    ece_val = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
    brier_val = float(brier_score_loss(y_true, y_prob)) if total > 0 else 0.0

    return {
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        },
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall_tpr": round(rec, 4),
        "specificity_tnr": round(spec, 4),
        "false_positive_rate": round(fpr_val, 4),
        "false_negative_rate": round(fnr_val, 4),
        "f1_score_synthetic": round(f1_syn, 4),
        "f1_score_real": round(f1_real, 4),
        "macro_f1": round(macro_f1, 4),
        "roc_auc": round(roc_auc_val, 4),
        "pr_auc": round(pr_auc_val, 4),
        "equal_error_rate": eer_val,
        "expected_calibration_error": ece_val,
        "brier_score": round(brier_val, 4)
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: str) -> None:
    """Plots and saves an annotated confusion matrix heatmap."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Forensic Confusion Matrix",
        ylabel="True Label (Ground Truth)",
        xlabel="Predicted Label"
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = val / max(1, cm.sum()) * 100.0
            ax.text(
                j, i, f"{val}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_roc_and_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, roc_path: str, pr_path: str) -> None:
    """Generates ROC and Precision-Recall evaluation curve plots."""
    os.makedirs(os.path.dirname(os.path.abspath(roc_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(pr_path)), exist_ok=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Chance Baseline")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate (FPR)")
    ax_roc.set_ylabel("True Positive Rate (TPR / Recall)")
    ax_roc.set_title("Receiver Operating Characteristic (ROC)")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_roc.savefig(roc_path, dpi=300)
    plt.close(fig_roc)

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    ax_pr.plot(recall, precision, color="teal", lw=2, label=f"PR Curve (AUC = {pr_auc:.4f})")
    baseline = np.mean(y_true)
    ax_pr.plot([0, 1], [baseline, baseline], color="red", lw=1.5, linestyle="--", label=f"Prevalence ({baseline:.2f})")
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend(loc="lower left")
    ax_pr.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_pr.savefig(pr_path, dpi=300)
    plt.close(fig_pr)


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str, n_bins: int = 10) -> None:
    """Generates reliability diagram for probability calibration."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", lw=2, color="purple", label="Model Calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability (Confidence)")
    ax.set_ylabel("Fraction of Positives (Empirical Accuracy)")
    ax.set_title("Reliability Diagram (Calibration Curve)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def extract_error_cases(
    manifest: List[Dict[str, Any]],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str = "reports/error_analysis.json",
    top_k: int = 10
) -> Dict[str, Any]:
    """Identifies and catalogs the top False Positives and False Negatives."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fps = []
    fns = []

    for idx, (label, prob) in enumerate(zip(y_true, y_prob)):
        item = manifest[idx] if idx < len(manifest) else {}
        entry = {
            "asset_id": item.get("asset_id", f"sample_{idx}"),
            "relative_path": item.get("relative_path", "unknown"),
            "generator_family": item.get("generator_family", "unknown"),
            "ground_truth": int(label),
            "predicted_prob": float(round(prob, 4))
        }

        # False positive: real image with high synthetic probability
        if label == 0 and prob >= 0.5:
            fps.append(entry)
        # False negative: synthetic image with low synthetic probability
        elif label == 1 and prob < 0.5:
            fns.append(entry)

    fps.sort(key=lambda x: x["predicted_prob"], reverse=True)
    fns.sort(key=lambda x: x["predicted_prob"])

    catalog = {
        "total_false_positives": len(fps),
        "total_false_negatives": len(fns),
        "worst_false_positives": fps[:top_k],
        "worst_false_negatives": fns[:top_k]
    }

    with open(output_path, "w") as f:
        json.dump(catalog, f, indent=2)

    return catalog


def evaluate_manifest(
    manifest_path: str,
    weights_path: str,
    config_path: str = "configs/preprocess.json",
    output_dir: str = "reports",
    output_prefix: str = ""
) -> Dict[str, Any]:
    """Executes end-to-end batch evaluation on a manifest file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    architecture = config.get("model_architecture", "efficientnet_b0")
    model = build_model(architecture=architecture, pretrained=False)
    load_checkpoint(weights_path, model, device=device)

    temp = float(config.get("calibration", {}).get("temperature", 1.0))
    dataloader = create_dataloader(manifest, batch_size=32, is_training=False, num_workers=0)

    all_targets = []
    all_calibrated_probs = []

    model.eval()
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs).cpu().numpy().flatten()
            y_batch = targets.numpy().flatten()

            calibrated = [calibrate_score(float(z), temperature=temp) for z in logits]
            all_calibrated_probs.extend(calibrated)
            all_targets.extend(y_batch)

    y_true_arr = np.array(all_targets)
    y_prob_arr = np.array(all_calibrated_probs)
    y_pred_arr = (y_prob_arr >= 0.50).astype(int)

    metrics = compute_all_metrics(y_true_arr, y_pred_arr, y_prob_arr)

    # Save metrics JSON
    metrics_file = os.path.join(output_dir, f"{output_prefix}metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate plots
    cm_arr = np.array([
        [metrics["confusion_matrix"]["true_negatives"], metrics["confusion_matrix"]["false_positives"]],
        [metrics["confusion_matrix"]["false_negatives"], metrics["confusion_matrix"]["true_positives"]]
    ])
    plot_confusion_matrix(cm_arr, ["Real", "Synthetic"], os.path.join(output_dir, f"{output_prefix}confusion_matrix.png"))
    plot_roc_and_pr_curves(
        y_true_arr, y_prob_arr,
        os.path.join(output_dir, f"{output_prefix}roc_curve.png"),
        os.path.join(output_dir, f"{output_prefix}pr_curve.png")
    )
    plot_calibration_curve(y_true_arr, y_prob_arr, os.path.join(output_dir, f"{output_prefix}calibration_curve.png"))

    # Extract error analysis
    extract_error_cases(manifest, y_true_arr, y_prob_arr, os.path.join(output_dir, f"{output_prefix}error_analysis.json"))

    print(f"Evaluation complete. Reports saved to '{output_dir}/'")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluation CLI harness.")
    parser.add_argument("--test_manifest", type=str, default="manifests/test_manifest.json")
    parser.add_argument("--weights", type=str, default="models/best_model.pth")
    parser.add_argument("--config", type=str, default="configs/preprocess.json")
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument("--output_prefix", type=str, default="")
    args = parser.parse_args()

    print(f"Running evaluation on '{args.test_manifest}'...")
    evaluate_manifest(
        manifest_path=args.test_manifest,
        weights_path=args.weights,
        config_path=args.config,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix
    )
