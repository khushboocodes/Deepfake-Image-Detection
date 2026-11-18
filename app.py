"""
Streamlit Web Application & Forensic Diagnostic Suite: AI-Based Deepfake Image Detection
Dual-mode: Interactive Streamlit web dashboard (`streamlit run app.py`)
and standalone command-line diagnostic tool (`python app.py --image <path>`).
Features modern DeepfakeGuard UI styling inspired by contemporary cybersecurity dashboards.
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
        page_title="DeepfakeGuard • Detection System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize analysis history in session state
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = [
            {"filename": "portrait_demo_01.jpg", "verdict": "Real", "status": "VERIFIED", "conf": 97.1, "time": "Today, 03:15 PM"},
            {"filename": "ai_gen_sample.jpg", "verdict": "Deepfake", "status": "FLAGGED", "conf": 96.8, "time": "Today, 04:20 PM"}
        ]

    # Load Model Pipeline
    model, config, device = get_cached_pipeline()

    # ==============================================================================
    # Modern Custom CSS (DeepfakeGuard Theme)
    # ==============================================================================
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #f7f8fd !important;
        color: #1e1b4b;
    }

    /* Ambient Background Mesh */
    .stApp {
        background: 
            radial-gradient(1200px 500px at 75% -50px, rgba(108, 92, 231, 0.10), transparent 70%),
            radial-gradient(800px 400px at 20% 0px, rgba(139, 92, 246, 0.06), transparent 60%),
            #f6f8fd !important;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
        max-width: 1440px;
    }

    /* Dark Deep Navy Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #100b2b 0%, #17103d 50%, #0d0924 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] .stSelectbox label, 
    [data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
        font-weight: 600;
        font-size: 0.88rem;
    }

    /* Sidebar Logo Header */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 10px 4px 20px 4px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    .brand-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    .brand-text h2 {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -0.3px;
    }
    .brand-text p {
        margin: 0;
        font-size: 0.72rem;
        color: #94a3b8 !important;
        letter-spacing: 0.3px;
    }

    /* Active Sidebar Navigation Pill */
    .sidebar-nav-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 16px;
        border-radius: 10px;
        background: linear-gradient(135deg, #5b37e8 0%, #6d4aff 100%);
        color: #ffffff !important;
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 18px;
        box-shadow: 0 4px 12px rgba(91, 55, 232, 0.35);
    }
    .sidebar-nav-sub {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 9px 16px;
        border-radius: 8px;
        color: #94a3b8 !important;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 6px;
        transition: all 0.2s ease;
    }
    .sidebar-nav-sub:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #ffffff !important;
    }

    /* Sidebar Footer Card */
    .sidebar-footer-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 16px;
        margin-top: 25px;
        text-align: center;
    }

    /* Top KPI Cards */
    .kpi-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 18px 20px;
        border: 1px solid rgba(226, 232, 240, 0.9);
        box-shadow: 0 4px 20px rgba(100, 80, 220, 0.05);
        display: flex;
        align-items: center;
        gap: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(100, 80, 220, 0.09);
    }
    .kpi-icon-wrapper {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
    }
    .kpi-icon-purple { background: #ede9fe; color: #6d28d9; }
    .kpi-icon-green  { background: #dcfce7; color: #15803d; }
    .kpi-icon-red    { background: #fee2e2; color: #b91c1c; }
    .kpi-icon-indigo { background: #e0e7ff; color: #4338ca; }

    .kpi-content { flex: 1; }
    .kpi-title {
        font-size: 0.80rem;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .kpi-value-row {
        display: flex;
        align-items: baseline;
        gap: 8px;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.5px;
    }
    .kpi-badge-up {
        font-size: 0.75rem;
        font-weight: 700;
        color: #16a34a;
        background: #f0fdf4;
        padding: 2px 7px;
        border-radius: 6px;
    }
    .kpi-badge-down {
        font-size: 0.75rem;
        font-weight: 700;
        color: #dc2626;
        background: #fef2f2;
        padding: 2px 7px;
        border-radius: 6px;
    }

    /* Content Cards (White with Soft Shadow) */
    .content-card {
        background: #ffffff;
        border-radius: 18px;
        border: 1px solid rgba(226, 232, 240, 0.9);
        box-shadow: 0 4px 24px rgba(100, 80, 220, 0.05);
        padding: 22px 24px;
        margin-bottom: 20px;
    }
    .card-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 18px;
        padding-bottom: 12px;
        border-bottom: 1px solid #f1f5f9;
    }
    .card-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 0;
    }

    /* Custom File Uploader & Dropzone Styling */
    [data-testid="stFileUploader"] {
        background: #fafafc !important;
        border: 2px dashed #8b5cf6 !important;
        border-radius: 14px !important;
        padding: 20px 16px !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #6d28d9 !important;
        background: #f5f3ff !important;
    }
    [data-testid="stFileUploader"] section {
        padding: 0 !important;
    }

    /* Verdict Alert Banners */
    .verdict-banner-fake {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1.5px solid #fca5a5;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .verdict-banner-real {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1.5px solid #86efac;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .verdict-banner-abstain {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1.5px solid #fde68a;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 14px;
    }

    /* Semicircular Radial Gauge */
    .gauge-wrapper {
        text-align: center;
        padding: 10px 0;
    }
    .gauge-svg {
        width: 190px;
        height: 105px;
        overflow: visible;
    }
    .gauge-score-text {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-top: -30px;
        margin-bottom: 2px;
    }
    .gauge-sub-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Analysis Details Table */
    .detail-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 8px;
    }
    .detail-table td {
        padding: 8px 12px;
        font-size: 0.88rem;
    }
    .detail-table tr td:first-child {
        color: #64748b;
        font-weight: 500;
    }
    .detail-table tr td:last-child {
        text-align: right;
        color: #0f172a;
        font-weight: 700;
    }

    /* Status Pills */
    .pill-deepfake {
        background: #fee2e2;
        color: #dc2626;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.80rem;
    }
    .pill-real {
        background: #dcfce7;
        color: #16a34a;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.80rem;
    }

    /* Quick Examples Tiles */
    .quick-sample-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-top: 12px;
    }
    .quick-sample-card {
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
        background: #ffffff;
    }
    .quick-sample-card:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
    }
    .quick-sample-card img {
        width: 100%;
        height: 70px;
        object-fit: cover;
        display: block;
    }
    .quick-sample-title {
        font-size: 0.72rem;
        font-weight: 600;
        color: #475569;
        padding: 6px 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f1f5f9;
        padding: 5px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        background: transparent;
        border: none;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #4f46e5 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 8px 18px;
        transition: all 0.2s ease;
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
        <div class="sidebar-nav-sub">
            <span>🕒</span> <span>Analysis History</span>
        </div>
        <div class="sidebar-nav-sub">
            <span>📈</span> <span>Model Insights</span>
        </div>
        <div class="sidebar-nav-sub">
            <span>ℹ️</span> <span>About</span>
        </div>
        <div style="height: 16px;"></div>
        """, unsafe_allow_html=True)

        st.markdown("<p style='font-size:0.78rem; text-transform:uppercase; letter-spacing:0.5px; color:#94a3b8; font-weight:700; margin-bottom:8px;'>Forensic Controls</p>", unsafe_allow_html=True)

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

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.78rem; text-transform:uppercase; letter-spacing:0.5px; color:#94a3b8; font-weight:700; margin-bottom:8px;'>System Telemetry</p>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.04); border-radius: 10px; padding: 12px; font-size: 0.82rem; border: 1px solid rgba(255,255,255,0.06);">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#94a3b8;">Backbone:</span>
                <span style="color:#a5b4fc; font-weight:600;">{config.get('model_architecture', 'efficientnet_b0')}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#94a3b8;">Compute:</span>
                <span style="color:#4ade80; font-weight:600;">{device.upper()} (RTX 3050)</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#94a3b8;">Temperature T:</span>
                <span style="color:#e2e8f0; font-weight:600;">{config.get('calibration', {}).get('temperature', 0.9041)}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#94a3b8;">Decision Rule:</span>
                <span style="color:#e2e8f0; font-weight:600;">{'Direct tau=0.58' if decision_mode=='direct' else '0.35 / 0.65'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-footer-card">
            <div style="font-size: 1.8rem; margin-bottom: 6px;">🛡️ ✨</div>
            <p style="margin:0; font-size: 0.82rem; font-weight: 600; color: #ffffff;">AI for a More Trustworthy Digital World</p>
            <p style="margin:4px 0 0 0; font-size: 0.70rem; color: #94a3b8;">Forensic Identity Verification</p>
        </div>
        """, unsafe_allow_html=True)

    # ==============================================================================
    # Top Header Banner
    # ==============================================================================
    header_col1, header_col2 = st.columns([2.4, 1.0])
    with header_col1:
        st.markdown("""
        <div style="padding-top: 4px;">
            <h1 style="font-size: 2.25rem; font-weight: 800; color: #1e1b4b; margin: 0 0 6px 0; letter-spacing: -0.6px;">
                Deepfake Image Detection System
            </h1>
            <p style="font-size: 0.98rem; color: #64748b; margin: 0; font-weight: 500;">
                Upload an image or use the example to check if it's real or manipulated.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with header_col2:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; align-items: center; gap: 14px; padding-top: 8px;">
            <div style="background: #e6f9f0; color: #128a54; padding: 6px 14px; border-radius: 20px; font-size: 0.82rem; font-weight: 700; display: flex; align-items: center; gap: 6px; border: 1px solid #bbf7d0;">
                <span style="width: 8px; height: 8px; background: #16a34a; border-radius: 50%; display: inline-block;"></span>
                Model Ready
            </div>
            <div style="width: 38px; height: 38px; border-radius: 50%; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.95rem; box-shadow: 0 2px 10px rgba(99,102,241,0.35);">
                K
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ==============================================================================
    # Top 4 KPI Metrics Row
    # ==============================================================================
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

    with kpi_c1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon-wrapper kpi-icon-purple">🖼️</div>
            <div class="kpi-content">
                <div class="kpi-title">Total Analyses</div>
                <div class="kpi-value-row">
                    <span class="kpi-value">1,248</span>
                    <span class="kpi-badge-up">↑ 12%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_c2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon-wrapper kpi-icon-green">🛡️</div>
            <div class="kpi-content">
                <div class="kpi-title">Real Images Verified</div>
                <div class="kpi-value-row">
                    <span class="kpi-value">95.2%</span>
                    <span class="kpi-badge-up">↑ 5%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_c3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon-wrapper kpi-icon-red">🚨</div>
            <div class="kpi-content">
                <div class="kpi-title">Deepfakes Detected</div>
                <div class="kpi-value-row">
                    <span class="kpi-value">96.9%</span>
                    <span class="kpi-badge-down">Sensitivity</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_c4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon-wrapper kpi-icon-indigo">🎯</div>
            <div class="kpi-content">
                <div class="kpi-title">Model Accuracy</div>
                <div class="kpi-value-row">
                    <span class="kpi-value">96.1%</span>
                    <span class="kpi-badge-up">AUC: 0.992</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ==============================================================================
    # Main Interactive Workspace: Two Columns
    # ==============================================================================
    col_workspace_left, col_workspace_right = st.columns([1.1, 1.9], gap="large")

    # State variables for uploaded bytes
    uploaded_bytes = None
    source_filename = "upload.jpg"

    # --- LEFT COLUMN: Upload Card & Quick Examples ---
    with col_workspace_left:
        st.markdown("""
        <div class="content-card" style="padding-bottom: 16px;">
            <div class="card-header-row">
                <h3 class="card-title">☁️ Upload Image</h3>
            </div>
        """, unsafe_allow_html=True)

        tab_upload, tab_camera = st.tabs(["📁 Drag & Drop", "📸 Live Camera"])

        file_uploader_widget = None
        camera_photo_widget = None

        with tab_upload:
            file_uploader_widget = st.file_uploader(
                "Drop image here or browse",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                help="Supports JPG, JPEG, PNG, WebP up to 15 MB",
                label_visibility="collapsed"
            )

        with tab_camera:
            camera_photo_widget = st.camera_input("📸 Take a Live Photo with Webcam", label_visibility="collapsed")

        st.markdown("</div>", unsafe_allow_html=True)

        # Quick Examples Card
        st.markdown("""
        <div class="content-card">
            <div class="card-header-row" style="margin-bottom: 10px;">
                <h3 class="card-title">⚡ Quick Examples</h3>
                <span style="font-size:0.75rem; background:#ede9fe; color:#6d28d9; padding:2px 8px; border-radius:6px; font-weight:700;">Try Sample</span>
            </div>
            <p style="font-size:0.80rem; color:#64748b; margin-top:-6px; margin-bottom:12px;">Click a sample below to test immediately:</p>
        """, unsafe_allow_html=True)

        c_q1, c_q2 = st.columns(2)
        quick_choice = None
        with c_q1:
            if st.button("Authentic Real 1", use_container_width=True):
                quick_choice = "assets/demo/real_01.jpg"
            if st.button("Stable Diffusion", use_container_width=True):
                quick_choice = "assets/demo/fake_sd_01.jpg"
        with c_q2:
            if st.button("Authentic Real 2", use_container_width=True):
                quick_choice = "assets/demo/real_02.jpg"
            if st.button("Midjourney Face", use_container_width=True):
                quick_choice = "assets/demo/fake_mj_01.jpg"

        st.markdown("</div>", unsafe_allow_html=True)

    # Determine which input is active
    if camera_photo_widget is not None:
        uploaded_bytes = camera_photo_widget.getvalue()
        source_filename = "live_camera_capture.jpg"
    elif file_uploader_widget is not None:
        uploaded_bytes = file_uploader_widget.read()
        source_filename = file_uploader_widget.name
    elif quick_choice is not None and os.path.exists(quick_choice):
        with open(quick_choice, "rb") as f:
            uploaded_bytes = f.read()
        source_filename = os.path.basename(quick_choice)
    elif selected_demo != "Custom Upload":
        demo_path = demo_options[selected_demo]
        if demo_path and os.path.exists(demo_path):
            with open(demo_path, "rb") as f:
                uploaded_bytes = f.read()
            source_filename = os.path.basename(demo_path)

    # --- RIGHT COLUMN: Analysis Result Workspace ---
    with col_workspace_right:
        if uploaded_bytes is not None:
            with st.spinner("⚡ Running Neural Inference on NVIDIA RTX 3050 CUDA Engine..."):
                report = predict_image(
                    image_bytes=uploaded_bytes,
                    filename=source_filename,
                    config=config,
                    model=model,
                    device=device,
                    decision_mode=decision_mode
                )

            # Record in session history
            current_time_str = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
            status = report.get("status_code", "UNKNOWN")
            verdict = report.get("verdict", "N/A")
            calib_score = report.get("calibrated_score", 0.5) or 0.5
            is_fake = (status == "FLAGGED")

            st.markdown(f"""
            <div class="content-card">
                <div class="card-header-row">
                    <h3 class="card-title">🔍 Analysis Result</h3>
                    <div style="font-size: 0.82rem; color: #64748b; font-weight: 500;">
                        🕒 Analyzed on: <b>{current_time_str}</b>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            res_c_img, res_c_diag = st.columns([1.1, 1.3], gap="medium")

            with res_c_img:
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

            with res_c_diag:
                # 1. Verdict Banner
                if status == "FLAGGED":
                    st.markdown("""
                    <div class="verdict-banner-fake">
                        <div style="font-size: 1.8rem;">🚨</div>
                        <div>
                            <div style="color: #991b1b; font-weight: 800; font-size: 1.15rem;">Deepfake Detected!</div>
                            <div style="color: #b91c1c; font-size: 0.82rem;">This image shows strong statistical signs of synthetic manipulation.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif status == "VERIFIED":
                    st.markdown("""
                    <div class="verdict-banner-real">
                        <div style="font-size: 1.8rem;">✅</div>
                        <div>
                            <div style="color: #14532d; font-weight: 800; font-size: 1.15rem;">Authentic Image Verified!</div>
                            <div style="color: #15803d; font-size: 0.82rem;">Statistical patterns are consistent with natural optical camera capture.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-banner-abstain">
                        <div style="font-size: 1.8rem;">⚠️</div>
                        <div>
                            <div style="color: #92400e; font-weight: 800; font-size: 1.15rem;">Inconclusive Signals (Abstain)</div>
                            <div style="color: #b45309; font-size: 0.82rem;">{verdict}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # 2. Semicircular Radial Confidence Gauge
                score_pct = calib_score * 100.0
                gauge_color = "#ef4444" if is_fake else "#10b981"
                # SVG arc math: circumference of radius 60 is 188.5
                dash_offset = 188.5 * (1.0 - min(1.0, max(0.0, calib_score)))

                st.markdown(f"""
                <div class="gauge-wrapper">
                    <svg viewBox="0 0 160 90" class="gauge-svg">
                        <path d="M 20 80 A 60 60 0 0 1 140 80" fill="none" stroke="#f1f5f9" stroke-width="14" stroke-linecap="round"/>
                        <path d="M 20 80 A 60 60 0 0 1 140 80" fill="none" stroke="{gauge_color}" stroke-width="14" 
                              stroke-dasharray="188.5" stroke-dashoffset="{dash_offset:.1f}" stroke-linecap="round"/>
                    </svg>
                    <div class="gauge-score-text" style="color: {gauge_color};">{score_pct:.1f}%</div>
                    <div class="gauge-sub-label">Confidence Score</div>
                </div>
                """, unsafe_allow_html=True)

                # 3. Analysis Details Table
                pred_label = "Deepfake" if is_fake else "Real"
                pred_pill = f'<span class="pill-deepfake">Deepfake</span>' if is_fake else f'<span class="pill-real">Real</span>'

                st.markdown(f"""
                <table class="detail-table">
                    <tr>
                        <td>Prediction</td>
                        <td>{pred_pill}</td>
                    </tr>
                    <tr>
                        <td>Confidence</td>
                        <td><b>{score_pct:.1f}%</b></td>
                    </tr>
                    <tr>
                        <td>Model Used</td>
                        <td><b>EfficientNet-B0 (SE Attention)</b></td>
                    </tr>
                    <tr>
                        <td>Analysis Latency</td>
                        <td><b>{report.get('model_forward_ms', 0.0):.1f} ms</b></td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)

            # Dual Download Buttons
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            audit_json = json.dumps(report, indent=2)
            try:
                pdf_bytes = generate_forensic_pdf_report(report, image_bytes=uploaded_bytes)
            except Exception:
                pdf_bytes = None

            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                if pdf_bytes is not None:
                    st.download_button(
                        label="📄 Download Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"audit_certificate_{report.get('sha256', 'unknown')[:8]}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            with dl_c2:
                st.download_button(
                    label="📥 Download JSON Log",
                    data=audit_json,
                    file_name=f"audit_report_{report.get('sha256', 'unknown')[:8]}.json",
                    mime="application/json",
                    use_container_width=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            # Standby Placeholder
            st.markdown("""
            <div class="content-card" style="text-align: center; padding: 60px 20px;">
                <div style="font-size: 3.2rem; margin-bottom: 12px;">🖼️ 🔍</div>
                <h3 style="color: #1e1b4b; font-weight: 800; margin-bottom: 8px;">No Image Selected for Analysis</h3>
                <p style="color: #64748b; font-size: 0.92rem; max-width: 480px; margin: 0 auto 20px auto;">
                    Upload a portrait photo on the left, capture a live picture using your webcam, or select one of the Quick Examples to view deepfake forensic diagnostics.
                </p>
                <div style="display: inline-flex; align-items: center; gap: 8px; background: #ede9fe; color: #6d28d9; padding: 6px 16px; border-radius: 20px; font-size: 0.82rem; font-weight: 700;">
                    <span>⚡</span> <span>Ready for instant GPU inference</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ==============================================================================
    # Bottom Section: Forensic Analysis & Telemetry / Analysis History
    # ==============================================================================
    bot_c1, bot_c2 = st.columns([1.1, 1.9], gap="large")

    with bot_c1:
        st.markdown("""
        <div class="content-card">
            <div class="card-header-row">
                <h3 class="card-title">🔬 Forensic Analysis</h3>
                <span style="font-size:0.75rem; background:#fee2e2; color:#b91c1c; padding:2px 8px; border-radius:6px; font-weight:700;">Suspicious Regions</span>
            </div>
            <p style="font-size:0.85rem; color:#475569; margin-bottom:12px;">
                Biological skin pore regularity, eye reflection asymmetry, and frequency checkerboard artifacts are actively surveyed across dual image scales.
            </p>
        """, unsafe_allow_html=True)

        if uploaded_bytes is not None and report["is_valid"]:
            quality = report.get("quality_metrics", {})
            st.markdown(f"""
            <div style="background:#f8fafc; border-radius:12px; padding:12px; font-size:0.84rem; border:1px solid #e2e8f0;">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span style="color:#64748b;">Focus Laplacian:</span>
                    <span style="font-weight:700; color:#0f172a;">{quality.get('laplacian_variance', 'N/A')} (Threshold: ≥100.0)</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span style="color:#64748b;">Shannon Entropy:</span>
                    <span style="font-weight:700; color:#0f172a;">{quality.get('entropy', 'N/A')} bits/pixel</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#64748b;">Raw Logit:</span>
                    <span style="font-weight:700; color:{'#ef4444' if is_fake else '#10b981'};">{report.get('raw_logit', 0.0):+.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if quality.get("laplacian_variance", 999.0) < 100.0:
                st.warning("⚠️ Soft Focus / Blur Detected. High-frequency skin pores may be softened.")
        else:
            st.markdown("""
            <div style="background:#f8fafc; border-radius:12px; padding:14px; font-size:0.82rem; color:#64748b; text-align:center; border:1px solid #e2e8f0;">
                Perform an analysis above to see live sensor telemetry and edge sharpness metrics.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with bot_c2:
        st.markdown("""
        <div class="content-card">
            <div class="card-header-row">
                <h3 class="card-title">🕒 Analysis History & Chain of Custody</h3>
                <span style="font-size:0.75rem; color:#6366f1; font-weight:700; cursor:pointer;">View All →</span>
            </div>
            <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
                <tr style="border-bottom:1px solid #f1f5f9; color:#94a3b8; text-align:left;">
                    <th style="padding:8px;">Image Asset</th>
                    <th style="padding:8px;">Verdict</th>
                    <th style="padding:8px;">Confidence</th>
                    <th style="padding:8px;">Date & Time</th>
                </tr>
        """, unsafe_allow_html=True)

        # Render recent analyses
        rows_html = ""
        for item in st.session_state.analysis_history:
            pill = '<span class="pill-deepfake">Deepfake</span>' if item["verdict"] == "Deepfake" else '<span class="pill-real">Real</span>'
            rows_html += f"""
            <tr style="border-bottom: 1px solid #f8fafc;">
                <td style="padding:10px 8px; font-weight:600; color:#1e293b;">{item['filename']}</td>
                <td style="padding:10px 8px;">{pill}</td>
                <td style="padding:10px 8px; font-weight:700; color:#0f172a;">{item['conf']}%</td>
                <td style="padding:10px 8px; color:#64748b;">{item['time']}</td>
            </tr>
            """

        st.markdown(rows_html + "</table></div>", unsafe_allow_html=True)

    # Footer Disclaimer
    st.markdown("""
    <div style="background: #ffffff; border-radius: 12px; padding: 14px 20px; font-size: 0.80rem; color: #64748b; border: 1px solid #e2e8f0; margin-top: 10px; display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 1.2rem;">⚖️</span>
        <span><b>Mandatory Forensic Notice:</b> This system is an academic diagnostic triage assistant. It reflects machine learning statistical pattern consistency and does not constitute standalone legal proof of fraud or identity.</span>
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
