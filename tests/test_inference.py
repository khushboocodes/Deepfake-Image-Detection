"""
Unit and Integration Tests for Calibration, Inference, and Decision Engine Module.
"""

import io
import pytest
import PIL.Image
from src.calibration import calibrate_score, fit_temperature
from src.predict import evaluate_decision, check_quality_abstention, predict_image
from src.report_pdf import generate_forensic_pdf_report
from src.model import build_model


def test_evaluate_decision_thresholds():
    thresholds = {"real_upper_bound": 0.35, "fake_lower_bound": 0.65}

    # Synthetic range (>= 0.65)
    v1, s1, b1 = evaluate_decision(0.85, thresholds)
    assert s1 == "FLAGGED"
    assert b1 == "red"

    # Real range (<= 0.35)
    v2, s2, b2 = evaluate_decision(0.15, thresholds)
    assert s2 == "VERIFIED"
    assert b2 == "green"

    # Uncertainty / Abstention range (0.35 - 0.65)
    v3, s3, b3 = evaluate_decision(0.50, thresholds)
    assert s3 == "ABSTAIN"
    assert b3 == "amber"


def test_evaluate_decision_direct_mode():
    thresholds = {"direct_threshold": 0.58}

    # Synthetic range (>= 0.58)
    v1, s1, b1 = evaluate_decision(0.60, thresholds, decision_mode="direct")
    assert s1 == "FLAGGED"
    assert b1 == "red"

    # Real range (< 0.58)
    v2, s2, b2 = evaluate_decision(0.50, thresholds, decision_mode="direct")
    assert s2 == "VERIFIED"
    assert b2 == "green"


def test_check_quality_abstention():
    # Low resolution (<128px)
    abstain_res, reason_res = check_quality_abstention({}, (100, 100))
    assert abstain_res is True
    assert "Resolution" in reason_res

    # Severe blur (Laplacian variance < 100)
    abstain_blur, reason_blur = check_quality_abstention({"laplacian_variance": 42.0}, (256, 256))
    assert abstain_blur is True
    assert "Focus" in reason_blur

    # Acceptable quality
    abstain_ok, reason_ok = check_quality_abstention({"laplacian_variance": 450.0}, (256, 256))
    assert abstain_ok is False
    assert reason_ok is None


def test_calibrate_score():
    # At logit = 0.0, sigmoid is exactly 0.5 regardless of T
    assert abs(calibrate_score(0.0, temperature=1.0) - 0.5) < 1e-3
    assert abs(calibrate_score(0.0, temperature=2.5) - 0.5) < 1e-3

    # Positive logit produces > 0.5
    assert calibrate_score(2.0, temperature=1.0) > 0.5
    # Negative logit produces < 0.5
    assert calibrate_score(-2.0, temperature=1.0) < 0.5


def test_predict_image_pipeline():
    model = build_model("efficientnet_b0", pretrained=False)
    config = {
        "model_architecture": "efficientnet_b0",
        "input_resolution": [224, 224],
        "calibration": {"temperature": 1.0},
        "decision_thresholds": {"real_upper_bound": 0.35, "fake_lower_bound": 0.65}
    }

    buf = io.BytesIO()
    PIL.Image.new("RGB", (256, 256), (180, 150, 120)).save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    report = predict_image(raw_bytes, "portrait.jpg", config, model, device="cpu")
    assert report["is_valid"] is True
    assert "sha256" in report
    assert "calibrated_score" in report
    assert report["status_code"] in ("FLAGGED", "VERIFIED", "ABSTAIN", "ABSTAIN_QUALITY")
    assert "disclaimer" in report


def test_pdf_report_generation():
    buf = io.BytesIO()
    PIL.Image.new("RGB", (200, 200), (120, 150, 180)).save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    mock_report = {
        "is_valid": True,
        "filename": "test_face.jpg",
        "sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        "format": "JPEG",
        "dimensions": [200, 200],
        "status_code": "VERIFIED",
        "verdict": "Consistent with Natural Optical Capture",
        "calibrated_score": 0.052,
        "raw_logit": -4.53,
        "model_forward_ms": 12.5,
        "total_pipeline_ms": 25.1,
        "quality_metrics": {"laplacian_variance": 312.4, "entropy": 7.55},
        "exif_summary": {"camera_make": "Sony", "camera_model": "ILCE-7M3", "software": "None", "datetime_original": "2024:05:01 12:00:00"},
        "disclaimer": "Test notice."
    }

    pdf_bytes = generate_forensic_pdf_report(mock_report, image_bytes=raw_bytes)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")

