"""
Unit and Integration Tests for Offline Evaluation and Metrics Suite Module.
"""

import os
import pytest
import numpy as np
from src.evaluate import (
    compute_all_metrics,
    compute_expected_calibration_error,
    compute_equal_error_rate,
    plot_confusion_matrix,
    extract_error_cases
)


def test_compute_all_metrics_known_values():
    # 4 samples: 2 real (0), 2 synthetic (1)
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])  # 1 TN, 1 FP, 1 FN, 1 TP
    y_prob = np.array([0.1, 0.8, 0.4, 0.9])

    metrics = compute_all_metrics(y_true, y_pred, y_prob)

    assert metrics["confusion_matrix"]["true_negatives"] == 1
    assert metrics["confusion_matrix"]["false_positives"] == 1
    assert metrics["confusion_matrix"]["false_negatives"] == 1
    assert metrics["confusion_matrix"]["true_positives"] == 1

    assert metrics["accuracy"] == 0.50
    assert metrics["balanced_accuracy"] == 0.50
    assert metrics["precision"] == 0.50
    assert metrics["recall_tpr"] == 0.50
    assert metrics["specificity_tnr"] == 0.50
    assert metrics["macro_f1"] == 0.50
    assert "roc_auc" in metrics
    assert "equal_error_rate" in metrics
    assert "expected_calibration_error" in metrics


def test_compute_expected_calibration_error():
    # Perfectly calibrated predictions: y_true == y_prob
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.05, 0.95, 0.95])
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 0.10


def test_extract_error_cases(tmp_path):
    manifest = [
        {"asset_id": "r1", "relative_path": "real1.jpg", "generator_family": "Real"},
        {"asset_id": "r2", "relative_path": "real2.jpg", "generator_family": "Real"},
        {"asset_id": "s1", "relative_path": "fake1.jpg", "generator_family": "SD"},
        {"asset_id": "s2", "relative_path": "fake2.jpg", "generator_family": "SD"},
    ]
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.85, 0.15, 0.9])  # r2 is FP (prob 0.85), s1 is FN (prob 0.15)

    out_file = str(tmp_path / "errors.json")
    catalog = extract_error_cases(manifest, y_true, y_prob, output_path=out_file, top_k=5)

    assert catalog["total_false_positives"] == 1
    assert catalog["total_false_negatives"] == 1
    assert catalog["worst_false_positives"][0]["asset_id"] == "r2"
    assert catalog["worst_false_negatives"][0]["asset_id"] == "s1"
    assert os.path.exists(out_file)
