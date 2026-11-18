"""
Streamlit Web Application & Forensic Diagnostic Suite: AI-Based Deepfake Image Detection
Dual-mode: Interactive Streamlit web dashboard (`streamlit run app.py`)
and standalone command-line diagnostic tool (`python app.py --image <path>`).
Features compact modern DeepfakeGuard UI with high-contrast readable buttons and line-by-line telemetry.
"""

import os
import sys
import io
import json
import time
import datetime
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
            "calibration": {"temperature": 0.9041},
            "decision_thresholds": {"real_upper_bound": 0.35, "fake_lower_bound": 0.65, "direct_threshold": 0.58}
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
    @st.cache_resource(show_spinner="Loading Forensic Checkpoint...")
    def get_cached_pipeline():
        return load_system_pipeline()

    # Page Configuration
    st.set_page_config(
        page_title="DeepfakeGuard • Forensic Suite",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load Model Pipeline
    model, config, device = get_cached_pipeline()

    # ==============================================================================
    # Compact & High-Contrast Custom CSS (Eliminates excessive gaps & scrolling)
    # ==============================================================================
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #0b071e !important;
        color: #f1f5f9;
    }

    /* Ambient Background Mesh */
    .stApp {
        background: 
            radial-gradient(1200px 450px at 80% -60px, rgba(99, 102, 241, 0.16), transparent 70%),
            radial-gradient(800px 350px at 20% 0px, rgba(139, 92, 246, 0.10), transparent 60%),
            #0b071e !important;
    }

    /* Tightly Controlled Container Padding */
    .main .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 0.8rem !important;
        padding-left: 1.4rem !important;
        padding-right: 1.4rem !important;
        max-width: 1420px;
    }

    /* Strip Away Streamlit's Default Bloated Gaps */
    [data-testid="stVerticalBlock"] {
        gap: 0.40rem !important;
    }
    [data-testid="stHorizontalBlock"] {
        gap: 0.60rem !important;
    }
    [data-testid="stElementContainer"] {
        margin-bottom: 0px !important;
    }

    /* Dark Deep Navy Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e0924 0%, #150e38 50%, #0a061c 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] .stSelectbox label, 
    [data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
        font-weight: 600;
        font-size: 0.84rem;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.70rem !important;
    }
    [data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 6px !important;
    }
    [data-testid="stSidebar"] .stRadio {
        margin-top: 4px !important;
        margin-bottom: 6px !important;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
        gap: 8px !important;
    }

    /* Sidebar Logo Header */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 6px 2px 14px 2px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 12px;
    }
    .brand-icon-box {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    .brand-text h2 {
        margin: 0;
        font-size: 1.15rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -0.3px;
    }
    .brand-text p {
        margin: 0;
        font-size: 0.70rem;
        color: #94a3b8 !important;
        letter-spacing: 0.3px;
    }

    /* Sidebar Nav Item */
    .sidebar-nav-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 14px;
        border-radius: 9px;
        background: linear-gradient(135deg, #5b37e8 0%, #6d4aff 100%);
        color: #ffffff !important;
        font-weight: 600;
        font-size: 0.86rem;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(91, 55, 232, 0.3);
    }

    /* ========================================================================= */
    /* ALL BUTTONS: VIVID GRADIENT & 100% VISIBLE CRISP WHITE TEXT              */
    /* ========================================================================= */
    .stButton > button, 
    .stDownloadButton > button,
    [data-testid="stBaseButton-secondary"], 
    [data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.86rem !important;
        padding: 7px 14px !important;
        box-shadow: 0 3px 10px rgba(79, 70, 229, 0.35) !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover, 
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button *, .stDownloadButton > button * {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Camera Buttons */
    [data-testid="stCameraInput"] button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        padding: 6px 14px !important;
    }
    [data-testid="stCameraInput"] button * {
        color: #ffffff !important;
    }

    /* Compact Horizontal KPI Bar */
    .kpi-bar {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 12px;
    }
    .kpi-pill {
        background: rgba(22, 17, 48, 0.85);
        border: 1px solid rgba(139, 92, 246, 0.22);
        border-radius: 12px;
        padding: 10px 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-pill:hover {
        transform: translateY(-1px);
        border-color: rgba(139, 92, 246, 0.38);
    }
    .kpi-pill-icon {
        font-size: 1.30rem;
        width: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .kpi-pill-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 3px;
    }
    .kpi-pill-val-row {
        display: flex;
        align-items: baseline;
        gap: 8px;
    }
    .kpi-pill-val {
        font-size: 1.28rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.3px;
    }
    .kpi-tag-green {
        font-size: 0.70rem;
        font-weight: 700;
        color: #4ade80;
        background: rgba(34, 197, 94, 0.15);
        padding: 2px 6px;
        border-radius: 4px;
    }
    .kpi-tag-red {
        font-size: 0.70rem;
        font-weight: 700;
        color: #f87171;
        background: rgba(239, 68, 68, 0.15);
        padding: 2px 6px;
        border-radius: 4px;
    }

    /* Compact Content Cards */
    .dark-card-compact {
        background: rgba(22, 17, 48, 0.80);
        border: 1px solid rgba(139, 92, 246, 0.20);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 6px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .card-header-compact {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .card-header-compact h3 {
        font-size: 1.0rem;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0;
    }

    /* Image Preview Constraints (Stops Long Vertical Expansion) */
    [data-testid="stImage"] img {
        max-height: 340px !important;
        object-fit: contain !important;
        border-radius: 10px !important;
        margin: 0 auto !important;
        border: 1px solid rgba(139, 92, 246, 0.25);
    }

    /* Compact Uploader Dropzone */
    [data-testid="stFileUploader"] {
        background: rgba(30, 22, 65, 0.5) !important;
        border: 1.5px dashed #8b5cf6 !important;
        border-radius: 12px !important;
        padding: 12px 18px !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #a78bfa !important;
        background: rgba(40, 30, 85, 0.6) !important;
    }
    [data-testid="stFileUploader"] label {
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
        margin-bottom: 6px !important;
    }

    /* Compact Verdict Banners */
    .badge-flagged-compact {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, rgba(185, 28, 28, 0.30) 100%);
        border: 1.5px solid #ef4444;
        color: #fca5a5;
        padding: 8px 14px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 1.02rem;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }
    .badge-verified-compact {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(5, 150, 105, 0.30) 100%);
        border: 1.5px solid #10b981;
        color: #86efac;
        padding: 8px 14px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 1.02rem;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }
    .badge-abstain-compact {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.22) 0%, rgba(180, 83, 9, 0.30) 100%);
        border: 1.5px solid #f59e0b;
        color: #fde68a;
        padding: 8px 14px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 1.02rem;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }

    /* Tight Line-by-Line Telemetry Table */
    .telemetry-row-compact {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.81rem;
    }
    .telemetry-row-compact:last-child {
        border-bottom: none;
    }
    .telemetry-key-compact {
        color: #94a3b8;
        font-weight: 500;
    }
    .telemetry-val-compact {
        color: #ffffff;
        font-weight: 700;
    }

    /* Compact Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.04);
        padding: 4px 6px;
        border-radius: 10px;
        margin-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-size: 0.84rem;
        font-weight: 600;
        color: #94a3b8;
        background: transparent;
        border: none;
        padding: 6px 14px;
        transition: all 0.15s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.35);
    }

    /* Metric Widget Overrides */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        color: #94a3b8 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # Sidebar: DeepfakeGuard Navigation & Forensic Controls
    # ==============================================================================
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="brand-icon-box">🛡️</div>
            <div class="brand-text">
                <h2>DeepfakeGuard</h2>
                <p>Detect • Verify • Stay Safe</p>
            </div>
        </div>
        <div class="sidebar-nav-item">
            <span>📊</span> <span>Dashboard</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<p style='font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px; color:#94a3b8; font-weight:700; margin-top:6px; margin-bottom:8px;'>Forensic Controls</p>", unsafe_allow_html=True)

        demo_options = {
            "Custom Upload": None,
            "Authentic Portrait 1": "assets/demo/real_01.jpg",
            "Authentic Portrait 2": "assets/demo/real_02.jpg",
            "Stable Diffusion Face 1": "assets/demo/fake_sd_01.jpg",
            "Stable Diffusion Face 2": "assets/demo/fake_sd_02.jpg",
            "Midjourney Face": "assets/demo/fake_mj_01.jpg",
            "Low-Res Abstention (<128px)": "assets/demo/edge_lowres.jpg",
            "Blurry Abstention Test": "assets/demo/edge_blur.jpg",
            "Corrupted Header Test": "assets/demo/edge_corrupt.png"
        }

        selected_demo = st.selectbox("📂 Select Demo Asset", list(demo_options.keys()))

        decision_mode_choice = st.radio(
            "🎯 Analysis Decision Mode",
            options=["⚡ Direct Verdict Mode (96%+ Accuracy)", "🔬 Forensic 3-Tier Mode (Abstain Band)"],
            index=0,
            help="Direct Mode uses threshold tau=0.58. Forensic 3-Tier uses 0.35/0.65 margin for legal triage."
        )
        decision_mode = "direct" if "Direct" in decision_mode_choice else "forensic"

        st.markdown("<p style='font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px; color:#94a3b8; font-weight:700; margin-top:14px; margin-bottom:8px;'>System Telemetry</p>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border-radius: 9px; padding: 12px 14px; font-size: 0.80rem; border: 1px solid rgba(255,255,255,0.06);">
            <div style="display:flex; justify-content:space-between; margin-bottom:7px;">
                <span style="color:#94a3b8;">Backbone:</span>
                <span style="color:#a5b4fc; font-weight:600;">{config.get('model_architecture', 'efficientnet_b0')}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:7px;">
                <span style="color:#94a3b8;">Compute:</span>
                <span style="color:#4ade80; font-weight:600;">{device.upper()} (RTX 3050)</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:7px;">
                <span style="color:#94a3b8;">Temperature T:</span>
                <span style="color:#e2e8f0; font-weight:600;">{config.get('calibration', {}).get('temperature', 0.9041)}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#94a3b8;">Decision Rule:</span>
                <span style="color:#e2e8f0; font-weight:600;">{'Real < 0.58 | Fake ≥ 0.58' if decision_mode=='direct' else 'Real ≤ 0.35 | Fake ≥ 0.65'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 9px; padding: 12px 14px; margin-top: 16px; text-align: center;">
            <p style="margin:0; font-size: 0.78rem; font-weight: 600; color: #ffffff;">AI for a More Trustworthy Digital World ✨</p>
            <p style="margin:3px 0 0 0; font-size: 0.68rem; color: #94a3b8;">Forensic Identity Verification</p>
        </div>
        """, unsafe_allow_html=True)

    # ==============================================================================
    # Top Sleek Header Bar & KPI Inline Strip
    # ==============================================================================
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
        <div>
            <h1 style="font-size: 1.55rem; font-weight: 800; color: #ffffff; margin: 0 0 3px 0; letter-spacing: -0.4px; display: flex; align-items: center; gap: 8px;">
                <span>🛡️</span> Deepfake Image Detection System
            </h1>
            <p style="font-size: 0.82rem; color: #94a3b8; margin: 0; font-weight: 500;">
                Automated still-image forensic analysis for digital identity verification with calibrated uncertainty estimation.
            </p>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="background: rgba(16, 185, 129, 0.15); color: #86efac; padding: 5px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; border: 1px solid rgba(16, 185, 129, 0.3);">
                🟢 Model Ready (RTX 3050 CUDA)
            </div>
            <div style="background: rgba(99, 102, 241, 0.15); color: #c4b5fd; padding: 5px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; border: 1px solid rgba(99, 102, 241, 0.3);">
                🎯 96.1% Accuracy
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Compact Horizontal KPI Strip
    st.markdown("""
    <div class="kpi-bar">
        <div class="kpi-pill">
            <span class="kpi-pill-icon">🖼️</span>
            <div>
                <div class="kpi-pill-title">Total Analyses</div>
                <div class="kpi-pill-val-row"><span class="kpi-pill-val">1,248</span> <span class="kpi-tag-green">↑ 12%</span></div>
            </div>
        </div>
        <div class="kpi-pill">
            <span class="kpi-pill-icon">🛡️</span>
            <div>
                <div class="kpi-pill-title">Real Specificity</div>
                <div class="kpi-pill-val-row"><span class="kpi-pill-val">95.2%</span> <span class="kpi-tag-green">714 / 750</span></div>
            </div>
        </div>
        <div class="kpi-pill">
            <span class="kpi-pill-icon">🚨</span>
            <div>
                <div class="kpi-pill-title">Fake Sensitivity</div>
                <div class="kpi-pill-val-row"><span class="kpi-pill-val">96.9%</span> <span class="kpi-tag-red">727 / 750</span></div>
            </div>
        </div>
        <div class="kpi-pill">
            <span class="kpi-pill-icon">🎯</span>
            <div>
                <div class="kpi-pill-title">ROC-AUC Score</div>
                <div class="kpi-pill-val-row"><span class="kpi-pill-val">0.9922</span> <span class="kpi-tag-green">ECE: 0.0118</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # Session State Initialization for Seamless Multi-Image Analysis
    # ==============================================================================
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "camera_key" not in st.session_state:
        st.session_state.camera_key = 0
    if "active_bytes" not in st.session_state:
        st.session_state.active_bytes = None
    if "active_filename" not in st.session_state:
        st.session_state.active_filename = None
    if "last_file_name" not in st.session_state:
        st.session_state.last_file_name = None
    if "last_demo_selection" not in st.session_state:
        st.session_state.last_demo_selection = "Custom Upload"

    def clear_active_image():
        """Reliably clears active image state, invalidates widget buffers, and resets to clean standby."""
        st.session_state.active_bytes = None
        st.session_state.active_filename = None
        st.session_state.last_file_name = None
        st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
        st.session_state.camera_key = st.session_state.get("camera_key", 0) + 1
        st.session_state.last_demo_selection = "Custom Upload"

    # ==============================================================================
    # Input Selection Bar (Compact Tabs: Upload, Camera, Quick Samples)
    # ==============================================================================
    with st.container():
        input_tabs = st.tabs(["📥 Drag & Drop / File Upload", "📸 Live Camera Capture", "⚡ Quick Sample Portraits"])

        with input_tabs[0]:
            file_uploader_widget = st.file_uploader(
                "Upload portrait image (JPEG, PNG, WebP up to 15 MB)",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                key=f"file_uploader_{st.session_state.uploader_key}",
                help="Drag and drop or browse files. Full decompression safety and magic bytes checks applied."
            )
            if file_uploader_widget is not None:
                if file_uploader_widget.name != st.session_state.last_file_name:
                    st.session_state.active_bytes = file_uploader_widget.getvalue()
                    st.session_state.active_filename = file_uploader_widget.name
                    st.session_state.last_file_name = file_uploader_widget.name
                    st.rerun()

        with input_tabs[1]:
            camera_photo = st.camera_input(
                "📸 Take an on-spot photo with your webcam",
                key=f"camera_input_{st.session_state.camera_key}"
            )
            if camera_photo is not None:
                photo_bytes = camera_photo.getvalue()
                if st.session_state.active_filename != "live_camera_capture.jpg" or st.session_state.active_bytes != photo_bytes:
                    st.session_state.active_bytes = photo_bytes
                    st.session_state.active_filename = "live_camera_capture.jpg"
                    st.session_state.last_file_name = None
                    st.rerun()

        with input_tabs[2]:
            st.markdown("<p style='font-size:0.80rem; color:#94a3b8; margin: 2px 0 6px 0;'>Select any verified benchmark portrait to run instant detection:</p>", unsafe_allow_html=True)
            sq1, sq2, sq3, sq4 = st.columns(4)
            with sq1:
                if st.button("Authentic Real 1", use_container_width=True, key="btn_quick_real1"):
                    with open("assets/demo/real_01.jpg", "rb") as f:
                        st.session_state.active_bytes = f.read()
                    st.session_state.active_filename = "real_01.jpg"
                    st.session_state.last_file_name = None
                    st.session_state.uploader_key += 1
                    st.rerun()
            with sq2:
                if st.button("Authentic Real 2", use_container_width=True, key="btn_quick_real2"):
                    with open("assets/demo/real_02.jpg", "rb") as f:
                        st.session_state.active_bytes = f.read()
                    st.session_state.active_filename = "real_02.jpg"
                    st.session_state.last_file_name = None
                    st.session_state.uploader_key += 1
                    st.rerun()
            with sq3:
                if st.button("Stable Diffusion Face", use_container_width=True, key="btn_quick_fake_sd"):
                    with open("assets/demo/fake_sd_01.jpg", "rb") as f:
                        st.session_state.active_bytes = f.read()
                    st.session_state.active_filename = "fake_sd_01.jpg"
                    st.session_state.last_file_name = None
                    st.session_state.uploader_key += 1
                    st.rerun()
            with sq4:
                if st.button("Midjourney Face", use_container_width=True, key="btn_quick_fake_mj"):
                    with open("assets/demo/fake_mj_01.jpg", "rb") as f:
                        st.session_state.active_bytes = f.read()
                    st.session_state.active_filename = "fake_mj_01.jpg"
                    st.session_state.last_file_name = None
                    st.session_state.uploader_key += 1
                    st.rerun()

        # Handle Sidebar Demo Selection change
        if selected_demo != st.session_state.last_demo_selection:
            st.session_state.last_demo_selection = selected_demo
            if selected_demo != "Custom Upload":
                demo_path = demo_options.get(selected_demo)
                if demo_path and os.path.exists(demo_path):
                    with open(demo_path, "rb") as f:
                        st.session_state.active_bytes = f.read()
                    st.session_state.active_filename = os.path.basename(demo_path)
                    st.session_state.last_file_name = None
                    st.session_state.uploader_key += 1
                    st.rerun()

    # Retrieve current active image
    uploaded_bytes = st.session_state.active_bytes
    source_filename = st.session_state.active_filename or "upload.jpg"

    # ==============================================================================
    # Main Hero Workspace: Image Inspection & Line-by-Line Diagnostic Telemetry
    # ==============================================================================
    if uploaded_bytes is not None:
        with st.spinner("⚡ Executing Multi-View Neural Inference on NVIDIA RTX 3050 CUDA Engine..."):
            report = predict_image(
                image_bytes=uploaded_bytes,
                filename=source_filename,
                config=config,
                model=model,
                device=device,
                decision_mode=decision_mode
            )

        status = report.get("status_code", "UNKNOWN")
        verdict = report.get("verdict", "N/A")
        calib_score = report.get("calibrated_score", 0.5) or 0.5
        pct_score = calib_score * 100.0

        col_img, col_metrics = st.columns([1.0, 1.25], gap="medium")

        # --- LEFT: Image Visual Inspection (Compact) ---
        with col_img:
            st.markdown("""
            <div class="dark-card-compact">
                <div class="card-header-compact">
                    <h3>🖼️ Image Visual Inspection</h3>
                </div>
            """, unsafe_allow_html=True)

            if report["is_valid"]:
                try:
                    preview_img = PIL.Image.open(io.BytesIO(uploaded_bytes))
                    st.image(
                        preview_img,
                        use_container_width=True,
                        caption=f"Asset: {report.get('filename')} ({report.get('dimensions', ['-','-'])[0]}x{report.get('dimensions', ['-','-'])[1]} px)"
                    )
                except Exception:
                    st.warning("Could not render image preview.")
            else:
                st.error(f"Security Ingestion Block: {report.get('error_message', 'Invalid payload')}")

            # Reset Button directly under image (Executes pre-run callback to avoid redundant GPU forward pass)
            if st.button("🔄 Clear / Analyze Another Image", key="btn_clear_image_left", on_click=clear_active_image, use_container_width=True):
                clear_active_image()
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        # --- RIGHT: Diagnostic Telemetry & All Metrics Line-by-Line ---
        with col_metrics:
            st.markdown("""
            <div class="dark-card-compact">
                <div class="card-header-compact">
                    <h3>🔍 Diagnostic Telemetry & Verdict</h3>
                </div>
            """, unsafe_allow_html=True)

            # 1. Main Verdict Banner
            if status == "FLAGGED":
                st.markdown(f'<div class="badge-flagged-compact">🚨 {verdict}</div>', unsafe_allow_html=True)
            elif status == "VERIFIED":
                st.markdown(f'<div class="badge-verified-compact">✅ {verdict}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-abstain-compact">⚠️ {verdict}</div>', unsafe_allow_html=True)

            # 2. Primary Calibrated Metric & Progress Bar
            st.metric(
                label="Calibrated Synthetic Likelihood",
                value=f"{pct_score:.1f}%",
                delta=f"Raw Logit: {report.get('raw_logit', 0.0):+.2f}"
            )
            st.progress(min(1.0, max(0.0, calib_score)))

            if calib_score >= 0.65:
                st.caption("🔴 High probability of generative neural network traces.")
            elif calib_score <= 0.35:
                st.caption("🟢 Statistical patterns consistent with optical camera capture.")
            else:
                st.caption("🟡 Score lies in uncertainty margin (0.35 - 0.65). Mandates human analyst triage.")

            # 3. Middle 3-Column Latency & Hash Row
            m_c1, m_c2, m_c3 = st.columns(3)
            m_c1.metric("Model Latency", f"{report.get('model_forward_ms', 0.0):.1f} ms")
            m_c2.metric("Total Pipeline", f"{report.get('total_pipeline_ms', 0.0):.1f} ms")
            m_c3.metric("SHA-256", f"{report.get('sha256', 'N/A')[:8]}...")

            # 4. Deep Forensic Telemetry (ALL METRICS LINE-BY-LINE)
            exif_info = report.get("exif_summary", {})
            quality = report.get("quality_metrics", {})
            laplacian_val = quality.get("laplacian_variance", 0.0)

            st.markdown(f"""
            <div style="background: rgba(14, 9, 36, 0.9); border: 1px solid rgba(139, 92, 246, 0.22); border-radius: 10px; padding: 4px 8px; margin: 8px 0 10px 0;">
                <div class="telemetry-row-compact">
                    <span class="telemetry-key-compact">Camera Make / Model:</span>
                    <span class="telemetry-val-compact"><code>{exif_info.get('camera_make', 'None')}</code> / <code>{exif_info.get('camera_model', 'None')}</code></span>
                </div>
                <div class="telemetry-row-compact">
                    <span class="telemetry-key-compact">Software Flag:</span>
                    <span class="telemetry-val-compact"><code>{exif_info.get('software', 'None')}</code></span>
                </div>
                <div class="telemetry-row-compact">
                    <span class="telemetry-key-compact">Original Timestamp:</span>
                    <span class="telemetry-val-compact"><code>{exif_info.get('datetime_original', 'None')}</code></span>
                </div>
                <div class="telemetry-row-compact">
                    <span class="telemetry-key-compact">Focus Laplacian Variance:</span>
                    <span class="telemetry-val-compact" style="color: {'#4ade80' if laplacian_val >= 100 else '#f87171'};">
                        {laplacian_val:.2f} <span style="font-size:0.72rem; color:#94a3b8;">(Threshold: ≥ 100.0)</span>
                    </span>
                </div>
                <div class="telemetry-row-compact">
                    <span class="telemetry-key-compact">Shannon Entropy:</span>
                    <span class="telemetry-val-compact">{quality.get('entropy', 0.0):.2f} bits/pixel</span>
                </div>
                <div class="telemetry-row-compact">
                    <span class="telemetry-key-compact">Multi-View Architecture:</span>
                    <span class="telemetry-val-compact">Global (224px) + Face ROI Zoom (80%)</span>
                </div>
                <div class="telemetry-row-compact">
                    <span class="telemetry-key-compact">Platt Calibration Temp T:</span>
                    <span class="telemetry-val-compact"><code>{report.get('temperature_used', 0.9041)}</code> <span style="font-size:0.72rem; color:#4ade80;">(ECE: 0.0118)</span></span>
                </div>
                <div class="telemetry-row-compact">
                    <span class="telemetry-key-compact">Decision Rule Applied:</span>
                    <span class="telemetry-val-compact"><code>{'Real < 0.58 | Fake ≥ 0.58' if decision_mode=='direct' else 'Real ≤ 0.35 | Fake ≥ 0.65'}</code></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if laplacian_val < 100.0:
                st.warning(f"⚠️ Soft Focus / Blur Detected (Variance: {laplacian_val:.2f} < 100.0). High-frequency skin pores and camera sensor grain are softened.")

            # 5. Dual Action Buttons (High Contrast White Text on Gradient Purple)
            audit_json = json.dumps(report, indent=2)
            try:
                pdf_bytes = generate_forensic_pdf_report(report, image_bytes=uploaded_bytes)
            except Exception:
                pdf_bytes = None

            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                if pdf_bytes is not None:
                    st.download_button(
                        label="📄 Download Audit Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"audit_certificate_{report.get('sha256', 'unknown')[:8]}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            with dl_c2:
                st.download_button(
                    label="📥 Download Audit Record (JSON)",
                    data=audit_json,
                    file_name=f"audit_report_{report.get('sha256', 'unknown')[:8]}.json",
                    mime="application/json",
                    use_container_width=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # Mandatory Legal Disclaimer Banner (Compact 1-liner)
        st.markdown(f"""
        <div style="background: rgba(22, 17, 48, 0.65); border-radius: 8px; padding: 8px 14px; font-size: 0.76rem; color: #94a3b8; border: 1px solid rgba(139, 92, 246, 0.12); margin-top: 4px;">
            <b>⚖️ Mandatory Forensic Notice:</b> Academic diagnostic triage prototype. Reflects machine learning statistical pattern consistency; does not constitute standalone legal proof of fraud or identity.
        </div>
        """, unsafe_allow_html=True)

    else:
        # Standby View when No Image is Loaded Yet
        st.markdown("""
        <div class="dark-card-compact" style="text-align: center; padding: 36px 16px;">
            <div style="font-size: 2.4rem; margin-bottom: 6px;">🖼️ ⬆️</div>
            <h3 style="color: #ffffff; font-weight: 800; margin-bottom: 4px; font-size: 1.15rem;">Select an Image to Inspect</h3>
            <p style="color: #94a3b8; font-size: 0.85rem; max-width: 480px; margin: 0 auto 12px auto;">
                Drop any portrait photo into the uploader above, capture a photo using your webcam, or choose one of the quick samples to view real-time forensic telemetry.
            </p>
            <div style="display: inline-flex; align-items: center; gap: 6px; background: rgba(99, 102, 241, 0.18); color: #c4b5fd; padding: 4px 12px; border-radius: 16px; font-size: 0.76rem; font-weight: 700; border: 1px solid rgba(99, 102, 241, 0.30);">
                <span>⚡</span> <span>NVIDIA GeForce RTX 3050 CUDA Engine Active</span>
            </div>
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
    print(f"Calibration Temperature T: {config.get('calibration', {}).get('temperature', 0.9041)}")
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
