"""
Inference, Decision Engine, and Latency Profiling Module
Executes safe ingestion, forensic metadata harvesting, deterministic preprocessing,
calibrated neural inference, three-tier decision banding, and forensic report compilation.
"""

import os
import sys
import time

# Ensure project root is in sys.path for direct CLI execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Tuple, Optional
import json
import argparse
import PIL.Image
import torch
import torch.nn as nn

from src.security import safe_ingest_image
from src.preprocess import (
    extract_forensic_metadata,
    normalize_exif_orientation,
    ensure_three_channel_rgb,
    resize_image_deterministic,
    image_to_normalized_tensor,
    compute_image_quality_metrics
)
from src.calibration import calibrate_score


def evaluate_decision(
    calibrated_score: float,
    thresholds: Optional[Dict[str, Any]] = None,
    decision_mode: str = "forensic"
) -> Tuple[str, str, str]:
    """
    Categorizes calibrated score into decisions:
    - Direct Mode: Score >= direct_threshold (default 0.58) -> FLAGGED, Score < direct_threshold -> VERIFIED
    - Forensic Mode: Three-Tier Bands (<= 0.35 Real, >= 0.65 Fake, else Abstain)
    
    Returns:
        Tuple: (verdict_string, status_code, badge_color)
    """
    direct_threshold = (thresholds or {}).get("direct_threshold", 0.58)
    if decision_mode.lower() == "direct":
        if calibrated_score >= direct_threshold:
            return "AI-Generated / Synthetic Traces Detected", "FLAGGED", "red"
        else:
            return "Consistent with Natural Optical Capture", "VERIFIED", "green"

    real_bound = (thresholds or {}).get("real_upper_bound", 0.35)
    fake_bound = (thresholds or {}).get("fake_lower_bound", 0.65)

    if calibrated_score >= fake_bound:
        return "AI-Generated / Synthetic Traces Detected", "FLAGGED", "red"
    elif calibrated_score <= real_bound:
        return "Consistent with Natural Optical Capture", "VERIFIED", "green"
    else:
        return "Inconclusive / Ambiguous Signals (Abstain)", "ABSTAIN", "amber"


def check_quality_abstention(
    quality_metrics: Dict[str, Any],
    original_dims: Tuple[int, int]
) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether input resolution or focus blur requires an automated quality abstention override.
    """
    w, h = original_dims
    if min(w, h) < 128:
        return True, "Low Resolution: Image dimension (<128px) lacks sufficient forensic texture detail."
    
    laplacian_var = quality_metrics.get("laplacian_variance", 999.0)
    if laplacian_var < 50.0:
        return True, "Focus / Compression Degradation: High-frequency texture is severely blurred or smoothed."
        
    return False, None


def predict_image(
    image_bytes: bytes,
    filename: str,
    config: Dict[str, Any],
    model: nn.Module,
    device: str = "cpu",
    decision_mode: str = "direct"
) -> Dict[str, Any]:
    """
    Executes end-to-end inference and returns a comprehensive forensic diagnostic report.
    """
    start_total_time = time.perf_counter()

    # Step 1: Security validation and safe ingest
    is_valid, pil_img, telemetry, error_msg = safe_ingest_image(image_bytes, filename, config)
    if not is_valid or pil_img is None:
        total_time_ms = (time.perf_counter() - start_total_time) * 1000.0
        return {
            "is_valid": False,
            "filename": (telemetry or {}).get("filename", filename),
            "sha256": (telemetry or {}).get("sha256", "N/A"),
            "status_code": "REJECTED_SECURITY",
            "verdict": "Validation Failed",
            "badge_color": "red",
            "error_message": error_msg,
            "calibrated_score": None,
            "raw_logit": None,
            "total_pipeline_ms": round(total_time_ms, 2)
        }

    # Step 2: Extract forensic metadata and quality analysis
    exif_metadata = extract_forensic_metadata(pil_img)
    quality_metrics = compute_image_quality_metrics(pil_img)
    orig_dims = tuple(telemetry["dimensions"])

    # Step 3: Check automated quality abstention override
    should_abstain_quality, quality_reason = check_quality_abstention(quality_metrics, orig_dims)

    # Step 4: Preprocessing pipeline matching training
    preprocessed_img = normalize_exif_orientation(pil_img)
    preprocessed_img = ensure_three_channel_rgb(preprocessed_img)

    target_res = tuple(config.get("input_resolution", [224, 224]))
    interpolation = config.get("interpolation", "bilinear")
    antialias = config.get("antialias", True)

    # Dual-Scale Multi-View Processing:
    # View 1: Full-context composition view
    resized_img = resize_image_deterministic(
        preprocessed_img,
        target_size=target_res,
        interpolation=interpolation,
        antialias=antialias
    )
    norm_cfg = config.get("normalization", {})
    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std = norm_cfg.get("std", [0.229, 0.224, 0.225])

    input_tensor_full = image_to_normalized_tensor(resized_img, mean=mean, std=std).to(device)

    # View 2: High-resolution Face-ROI crop (zoom to central 80% to preserve biological skin pores & textures)
    w_orig, h_orig = preprocessed_img.size
    crop_box = (int(0.08 * w_orig), int(0.05 * h_orig), int(0.92 * w_orig), int(0.90 * h_orig))
    face_crop_img = preprocessed_img.crop(crop_box)
    face_crop_resized = resize_image_deterministic(
        face_crop_img,
        target_size=target_res,
        interpolation=interpolation,
        antialias=antialias
    )
    input_tensor_crop = image_to_normalized_tensor(face_crop_resized, mean=mean, std=std).to(device)

    # Step 5: Neural network forward pass on both views & latency measurement
    model.eval()
    start_infer_time = time.perf_counter()
    with torch.no_grad():
        raw_logit_full = float(model(input_tensor_full).cpu().numpy().flatten()[0])
        raw_logit_crop = float(model(input_tensor_crop).cpu().numpy().flatten()[0])
    model_forward_ms = (time.perf_counter() - start_infer_time) * 1000.0

    # Multi-Scale Forensic Fusion:
    # 1. If full view catches strong generative/boundary warping (>= 1.0), flag fake!
    # 2. If face crop confirms natural human skin pores / biological details (<= -1.2), verify real!
    # 3. Otherwise, balanced combination
    if raw_logit_full >= 1.0:
        raw_logit = raw_logit_full
    elif raw_logit_crop <= -1.2:
        raw_logit = raw_logit_crop
    else:
        raw_logit = 0.5 * raw_logit_full + 0.5 * raw_logit_crop

    # Step 6: Post-processing temperature calibration
    temp = float(config.get("calibration", {}).get("temperature", 1.0))
    calibrated_score = calibrate_score(raw_logit, temperature=temp)

    # Step 7: Decision engine evaluation
    thresholds = config.get("decision_thresholds", {})
    if should_abstain_quality:
        verdict = f"Abstain: {quality_reason}"
        status_code = "ABSTAIN_QUALITY"
        badge_color = "amber"
    else:
        verdict, status_code, badge_color = evaluate_decision(calibrated_score, thresholds, decision_mode=decision_mode)

    total_time_ms = (time.perf_counter() - start_total_time) * 1000.0

    disclaimer = (
        "Forensic Diagnostic Notice: This score reflects machine learning statistical pattern consistency. "
        "It does NOT constitute legal proof of authenticity or fraud. Missing EXIF metadata does not indicate fraud, "
        "and absence of synthetic traces does not prove real-world identity or physical liveness."
    )

    return {
        "is_valid": True,
        "filename": telemetry["filename"],
        "sha256": telemetry["sha256"],
        "format": telemetry["format"],
        "dimensions": telemetry["dimensions"],
        "filesize_bytes": telemetry["filesize_bytes"],
        "exif_summary": exif_metadata,
        "quality_metrics": quality_metrics,
        "raw_logit": round(raw_logit, 4),
        "temperature_used": temp,
        "calibrated_score": calibrated_score,
        "verdict": verdict,
        "status_code": status_code,
        "badge_color": badge_color,
        "model_forward_ms": round(model_forward_ms, 2),
        "total_pipeline_ms": round(total_time_ms, 2),
        "device_used": device,
        "disclaimer": disclaimer
    }


if __name__ == "__main__":
    from src.model import build_model, load_checkpoint

    parser = argparse.ArgumentParser(description="Deepfake Image Forensic Diagnostic CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to image file for diagnosis")
    parser.add_argument("--config", type=str, default="configs/preprocess.json", help="Path to preprocess config")
    parser.add_argument("--weights", type=str, default="models/best_model.pth", help="Path to trained model weights")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Target image file '{args.image}' not found.")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch = config.get("model_architecture", "efficientnet_b0")
    model = build_model(architecture=arch, pretrained=False)
    load_checkpoint(args.weights, model, device=device)
    model.eval()

    with open(args.image, "rb") as f:
        image_bytes = f.read()

    result = predict_image(image_bytes, os.path.basename(args.image), config, model, device=device)

    print("\n" + "=" * 60)
    print("      DEEPFAKE IMAGE FORENSIC DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"Image File:        {result.get('filename')}")
    print(f"Verdict:           {result.get('verdict')}")
    print(f"Status Code:       {result.get('status_code')}")
    print(f"Calibrated Score:  {result.get('calibrated_score')}  (0.0 = Authentic, 1.0 = AI Synthetic)")
    print(f"Raw Logit:         {result.get('raw_logit')}")
    print(f"Latency:           {result.get('total_pipeline_ms')} ms")
    if result.get("quality_metrics"):
        qm = result["quality_metrics"]
        print(f"Quality Metrics:   Blur Var={qm.get('blur_variance')} | Shannon Entropy={qm.get('entropy')}")
    print("=" * 60)
    print(f"Notice: {result.get('disclaimer')}\n")
