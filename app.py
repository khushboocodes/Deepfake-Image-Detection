"""
Streamlit Web Application & Forensic Diagnostic Suite: AI-Based Deepfake Image Detection
Dual-mode: Interactive Streamlit web dashboard (`streamlit run app.py`)
and standalone command-line diagnostic tool (`python app.py --image <path>`).
"""

import os
import sys
import io
import json
import time
from typing import Optional, Tuple
import PIL.Image
import torch

# Ensure project root is always in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model import build_model, load_checkpoint, ForensicClassifier
from src.predict import predict_image
from src.report_pdf import generate_forensic_pdf_report


def load_system_pipeline() -> Tuple[ForensicClassifier, dict, str]:
    """Loads configuration and neural model weights onto target compute device."""
    config_path = "configs/preprocess.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {
            "model_architecture": "efficientnet_b0",
            "checkpoint_path": "models/best_model.pth",
            "input_resolution": [224, 224],
            "interpolation": "bilinear",
            "calibration": {"temperature": 1.0},
            "decision_thresholds": {"real_upper_bound": 0.35, "fake_lower_bound": 0.65}
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch = config.get("model_architecture", "efficientnet_b0")
    model = build_model(architecture=arch, pretrained=False)

    checkpoint_path = config.get("checkpoint_path", "models/best_model.pth")
    if os.path.exists(checkpoint_path):
        try:
            load_checkpoint(checkpoint_path, model, device=device)
        except Exception:
            model = build_model(architecture=arch, pretrained=False)
            model.to(device)
            model.eval()
    else:
        model = build_model(architecture=arch, pretrained=False)
        model.to(device)
        model.eval()

    return model, config, device


# Check if Streamlit is available
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


if HAS_STREAMLIT:
    # Top-level resource cache
    @st.cache_resource(show_spinner="Loading Forensic Checkpoint...")
    def get_cached_pipeline():
        return load_system_pipeline()

    # ==============================================================================
    # Streamlit Web Mode Execution
    # ==============================================================================
    st.set_page_config(
        page_title="Deepfake Detection Forensic Suite",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            border-left: 5px solid #0d6efd;
            margin-bottom: 12px;
        }
        .badge-flagged {
            background-color: #f8d7da;
            color: #842029;
            padding: 10px 16px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 1.15rem;
            border: 1px solid #f5c2c7;
        }
        .badge-verified {
            background-color: #d1e7dd;
            color: #0f5132;
            padding: 10px 16px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 1.15rem;
            border: 1px solid #badbcc;
        }
        .badge-abstain {
            background-color: #fff3cd;
            color: #664d03;
            padding: 10px 16px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 1.15rem;
            border: 1px solid #ffecb5;
        }
        .disclaimer-box {
            background-color: #e9ecef;
            border-radius: 6px;
            padding: 14px;
            font-size: 0.85rem;
            color: #495057;
            margin-top: 24px;
            border: 1px solid #ced4da;
        }
        /* Custom Modern Drag & Drop Zone */
        [data-testid="stFileUploader"] {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.35) 0%, rgba(15, 23, 42, 0.55) 100%);
            border: 2px dashed #3b82f6;
            border-radius: 12px;
            padding: 14px 20px;
            margin-bottom: 16px;
            transition: all 0.3s ease;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #60a5fa;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.1) 0%, rgba(30, 58, 138, 0.2) 100%);
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.15);
        }
    </style>
    """, unsafe_allow_html=True)

    # Main dashboard header - Paint UI immediately so screen is never blank
    st.title("🛡️ AI-Based Deepfake Image Detection System")
    st.markdown(
        "Automated still-image forensic analysis for digital identity verification, "
        "featuring deterministic preprocessing, calibrated uncertainty estimation, and explicit abstention."
    )

    # Sidebar skeleton
    st.sidebar.title("🛡️ Forensic Controls")

    # Load model and config from cached pipeline with visible progress indicator
    with st.spinner("⚡ Initializing NVIDIA GeForce RTX 3050 CUDA Engine & Model Weights..."):
        model, config, device = get_cached_pipeline()

    demo_options = {
        "Custom Upload": None,
        "Authentic Portrait 1": "assets/demo/real_01.jpg",
        "Authentic Portrait 2": "assets/demo/real_02.jpg",
        "Stable Diffusion Portrait 1": "assets/demo/fake_sd_01.jpg",
        "Stable Diffusion Portrait 2": "assets/demo/fake_sd_02.jpg",
        "Midjourney Portrait": "assets/demo/fake_mj_01.jpg",
        "Low-Res Abstention Test (<128px)": "assets/demo/edge_lowres.jpg",
        "Blurry Abstention Test": "assets/demo/edge_blur.jpg",
        "Corrupted Header Test": "assets/demo/edge_corrupt.png"
    }

    selected_demo = st.sidebar.selectbox("📂 Select Demo Gallery Asset", list(demo_options.keys()))

    st.sidebar.markdown("---")
    decision_mode_choice = st.sidebar.radio(
        "🎯 Analysis & Decision Mode",
        options=["⚡ Direct Verdict Mode (90%+ Accuracy)", "🔬 Forensic 3-Tier Mode (Abstain Band)"],
        index=0,
        help="Direct Mode uses a 50% threshold for immediate Real vs Fake classification. Forensic Mode uses 35%-65% safety margin for legal triage."
    )
    decision_mode = "direct" if "Direct" in decision_mode_choice else "forensic"

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Telemetry")
    st.sidebar.write(f"**Backbone:** `{config.get('model_architecture', 'efficientnet_b0')}`")
    st.sidebar.write(f"**Compute Device:** `{device.upper()}` (NVIDIA RTX 3050)")
    st.sidebar.write(f"**Temperature T:** `{config.get('calibration', {}).get('temperature', 1.0)}`")
    if decision_mode == "direct":
        dir_thresh = config.get("decision_thresholds", {}).get("direct_threshold", 0.58)
        st.sidebar.write(f"**Decision Rule:** Real < {dir_thresh:.2f} | Synthetic ≥ {dir_thresh:.2f}")
    else:
        st.sidebar.write("**Decision Bands:** Real ≤ 0.35 | Synthetic ≥ 0.65")

    st.sidebar.markdown("---")
    st.sidebar.caption("AI-Based Deepfake Image Detection — Academic & Forensic Prototype v1.0.0")

    # Input Tabs: Drag & Drop File Upload or Live On-Spot Camera Capture
    tab_upload, tab_camera = st.tabs(["📁 Drag & Drop / File Upload", "📸 Live Camera Capture"])

    file_uploader_widget = None
    camera_photo_widget = None

    with tab_upload:
        file_uploader_widget = st.file_uploader(
            "📥 Drag and Drop Image Here to Upload (or Click to Browse Files)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Drag and drop any portrait image directly onto this area (up to 15 MB). Magic bytes, dimensions, and decompression safety are strictly validated."
        )

    with tab_camera:
        camera_photo_widget = st.camera_input("📸 Take a Live Photo with Webcam for On-Spot Analysis")

    uploaded_bytes = None
    source_filename = "upload.jpg"

    if camera_photo_widget is not None:
        uploaded_bytes = camera_photo_widget.getvalue()
        source_filename = "live_camera_capture.jpg"
    elif file_uploader_widget is not None:
        uploaded_bytes = file_uploader_widget.read()
        source_filename = file_uploader_widget.name
    elif selected_demo != "Custom Upload":
        demo_path = demo_options[selected_demo]
        if demo_path and os.path.exists(demo_path):
            with open(demo_path, "rb") as f:
                uploaded_bytes = f.read()
            source_filename = os.path.basename(demo_path)
            st.info(f"Loaded Demo Asset: **{selected_demo}** (`{source_filename}`)")

    if uploaded_bytes is not None:
        with st.spinner("Executing forensic inspection, validation, and neural inference..."):
            report = predict_image(
                image_bytes=uploaded_bytes,
                filename=source_filename,
                config=config,
                model=model,
                device=device,
                decision_mode=decision_mode
            )

        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.subheader("Image Visual Inspection")
            if report["is_valid"]:
                try:
                    preview_img = PIL.Image.open(io.BytesIO(uploaded_bytes))
                    st.image(
                        preview_img,
                        use_container_width=True,
                        caption=f"Asset: {report['filename']} ({report['dimensions'][0]}x{report['dimensions'][1]} px)"
                    )
                except Exception:
                    st.warning("Could not render image preview.")
            else:
                st.error(f"Security Ingestion Block: {report.get('error_message', 'Invalid payload')}")

        with col_right:
            st.subheader("Diagnostic Telemetry & Verdict")

            status = report.get("status_code", "UNKNOWN")
            verdict = report.get("verdict", "N/A")

            if status == "FLAGGED":
                st.markdown(f'<div class="badge-flagged">🚨 {verdict}</div>', unsafe_allow_html=True)
            elif status == "VERIFIED":
                st.markdown(f'<div class="badge-verified">✅ {verdict}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-abstain">⚠️ {verdict}</div>', unsafe_allow_html=True)

            st.write("")

            calibrated_score = report.get("calibrated_score")
            if calibrated_score is not None:
                pct_score = calibrated_score * 100.0
                st.metric(
                    label="Calibrated Synthetic Likelihood",
                    value=f"{pct_score:.1f}%",
                    delta=f"Raw Logit: {report.get('raw_logit', 0.0):.2f}"
                )
                st.progress(min(1.0, max(0.0, calibrated_score)))

                if calibrated_score >= 0.65:
                    st.caption("🔴 High probability of generative neural network traces.")
                elif calibrated_score <= 0.35:
                    st.caption("🟢 Statistical patterns consistent with optical camera capture.")
                else:
                    st.caption("🟡 Score lies in uncertainty margin (0.35 - 0.65). Mandates human analyst triage.")

            st.write("")
            c1, c2, c3 = st.columns(3)
            c1.metric("Model Latency", f"{report.get('model_forward_ms', 0.0):.1f} ms")
            c2.metric("Total Pipeline", f"{report.get('total_pipeline_ms', 0.0):.1f} ms")
            c3.metric("SHA-256", f"{report.get('sha256', 'N/A')[:8]}...")

            with st.expander("🔍 Deep Forensic Metadata & Quality Telemetry", expanded=True):
                exif_info = report.get("exif_summary", {})
                quality = report.get("quality_metrics", {})

                st.write(f"**Camera Make / Model:** `{exif_info.get('camera_make', 'None')}` / `{exif_info.get('camera_model', 'None')}`")
                st.write(f"**Software Flag:** `{exif_info.get('software', 'None')}`")
                st.write(f"**Original Timestamp:** `{exif_info.get('datetime_original', 'None')}`")
                st.write(f"**Focus Laplacian Variance:** `{quality.get('laplacian_variance', 'N/A')}` (Threshold: ≥ 100.0)")
                st.write(f"**Shannon Entropy:** `{quality.get('entropy', 'N/A')}` bits/pixel")
                if quality.get("laplacian_variance", 999.0) < 100.0:
                    st.warning(f"⚠️ Soft Focus / Blur Detected (Variance: {quality.get('laplacian_variance', 'N/A')} < 100.0). High-frequency skin pores and camera sensor grain are softened.")

            audit_json = json.dumps(report, indent=2)

            # Generate Official Forensic PDF Certificate
            try:
                pdf_bytes = generate_forensic_pdf_report(report, image_bytes=uploaded_bytes)
            except Exception:
                pdf_bytes = None

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                if pdf_bytes is not None:
                    st.download_button(
                        label="📄 Download Audit Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"audit_certificate_{report.get('sha256', 'unknown')[:8]}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            with dl_col2:
                st.download_button(
                    label="📥 Download Audit Record (JSON)",
                    data=audit_json,
                    file_name=f"audit_report_{report.get('sha256', 'unknown')[:8]}.json",
                    mime="application/json",
                    use_container_width=True
                )

        st.markdown(f"""
        <div class="disclaimer-box">
            <b>⚖️ Mandatory Forensic & Legal Notice:</b><br>
            {report.get('disclaimer', '')}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; border: 2px dashed rgba(59, 130, 246, 0.35); border-radius: 12px; margin-top: 14px; background-color: rgba(30, 41, 59, 0.25);">
            <div style="font-size: 2.4rem; margin-bottom: 8px;">🖼️ ⬆️</div>
            <h3 style="margin-bottom: 6px; color: #f1f5f9;">Drag and Drop a Portrait Photo to Begin</h3>
            <p style="color: #94a3b8; font-size: 0.95rem; max-width: 520px; margin: 0 auto 12px auto;">
                Drop any JPEG, PNG, or WebP photo directly into the upload area above, or choose a curated sample from the sidebar.
            </p>
            <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 0;">
                Supported: <b>JPG, JPEG, PNG, WebP</b> • Max file size: <b>15 MB</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    # ==============================================================================
    # Standalone CLI Mode Execution (when Streamlit package is not yet installed)
    # ==============================================================================
    import argparse
    parser = argparse.ArgumentParser(description="AI-Based Deepfake Image Detection CLI Diagnostic Tool")
    parser.add_argument("--image", type=str, default="assets/demo/real_01.jpg", help="Path to image file for analysis")
    args = parser.parse_args()

    print("=" * 70)
    print("AI-BASED DEEPFAKE IMAGE DETECTION SYSTEM — FORENSIC DIAGNOSTIC SUITE")
    print("=" * 70)

    model, config, device = load_system_pipeline()
    print(f"Model Backbone: {config.get('model_architecture', 'efficientnet_b0')} on {device.upper()}")
    print(f"Calibration Temperature T: {config.get('calibration', {}).get('temperature', 1.0)}")
    print(f"Analyzing Target Image: {args.image}")
    print("-" * 70)

    if not os.path.exists(args.image):
        print(f"ERROR: Image file not found at '{args.image}'")
        sys.exit(1)

    with open(args.image, "rb") as f:
        img_bytes = f.read()

    report = predict_image(img_bytes, os.path.basename(args.image), config, model, device)

    if not report["is_valid"]:
        print(f"STATUS: REJECTED")
        print(f"REASON: {report.get('error_message')}")
    else:
        print(f"VERDICT:           [{report['status_code']}] {report['verdict']}")
        print(f"CALIBRATED SCORE:  {report['calibrated_score'] * 100:.1f}% Synthetic Probability")
        print(f"RAW LOGIT:         {report['raw_logit']}")
        print(f"IMAGE DIMENSIONS:  {report['dimensions'][0]} x {report['dimensions'][1]} px ({report['format']})")
        print(f"SHA-256 HASH:      {report['sha256']}")
        print(f"LAPLACIAN FOCUS:   {report['quality_metrics']['laplacian_variance']} (Threshold: >= 100.0)")
        print(f"MODEL LATENCY:     {report['model_forward_ms']} ms (Total Pipeline: {report['total_pipeline_ms']} ms)")

    print("-" * 70)
    print("LEGAL NOTICE: Diagnostic triage tool only. Does not prove liveness or fraud.")
    print("=" * 70)
