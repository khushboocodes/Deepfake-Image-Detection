# 🛡️ AI-Based Deepfake Image Detection System for Digital Identity Verification

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.14-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4%20(RTX%203050)-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Test Suite](https://img.shields.io/badge/PyTest-29%20Passed%20(100%25)-success.svg)](file:///tests/)

An enterprise-grade, academically defensible, and reproducible still-image forensic verification suite engineered for **Digital Identity Verification (KYC/AML, remote onboarding, credential fraud auditing)**. 

The system couples a transfer-learning convolutional neural network (**EfficientNet-B0 with Squeeze-and-Excitation attention**) with defensive file ingestion, deterministic color/orientation normalization, **Dual-Scale Multi-View Inference (Global Composition + Central 80% Face ROI Zoom)**, **Platt Temperature Scaling ($T=0.9041$)**, a **Dual-Mode Decision Engine (Direct Mode $\tau=0.58$ vs Forensic 3-Tier Mode)**, and automated **Executive PDF Audit Certificate generation**.

---

## 📑 Table of Contents
1. [Executive Summary & Verified Benchmarks](#-executive-summary--verified-benchmarks)
2. [Key Capabilities & Features](#-key-capabilities--features)
3. [Complete Directory & File Architecture](#-complete-directory--file-architecture)
4. [Quickstart: Setup & Execution Guide](#-quickstart-setup--execution-guide)
5. [How to Use the Interactive Dashboard](#-how-to-use-the-interactive-dashboard)
6. [Complete Dashboard Metrics & Telemetry Guide](#-complete-dashboard-metrics--telemetry-guide)
7. [System Architecture & Processing Pipeline](#-system-architecture--processing-pipeline)
8. [Offline Model Training & Retraining Protocol](#-offline-model-training--retraining-protocol)
9. [Operational Troubleshooting & Common Scenarios](#-operational-troubleshooting--common-scenarios)
10. [Legal, Ethical & Forensic Disclaimers](#-legal-ethical--forensic-disclaimers)

---

## 📊 Executive Summary & Verified Benchmarks

The system was trained on a balanced benchmark of **10,000 curated portrait images** (5,000 Real from FFHQ vs 5,000 Synthetic from StyleGAN2 / Kaggle 140k dataset) using a dedicated **NVIDIA GeForce RTX 3050 Laptop GPU** (6GB VRAM, CUDA 12.4).

### Final Locked Test Split Performance (1,500 Held-Out Images):
* **Overall Test Accuracy:** **96.07%** (1,441 / 1,500 correct classifications)
* **ROC-AUC Score:** **0.9922** (Area Under the Receiver Operating Characteristic Curve)
* **PR-AUC Score:** **0.9929** (Precision-Recall Area Under Curve)
* **Synthetic Sensitivity / Recall:** **96.93%** (727 out of 750 fake images correctly flagged)
* **Real Specificity / TNR:** **95.20%** (714 out of 750 authentic images verified real)
* **False Alarm Rate (FPR):** **4.80%** (Low risk of falsely rejecting authentic users)
* **Expected Calibration Error (ECE):** **0.0118** (Well-calibrated probabilities)
* **Model Inference Latency:** **~140 ms** on GPU (Total pipeline: **~190 ms**)

```
                      PREDICTED REAL        PREDICTED SYNTHETIC
ACTUAL REAL (750)     714 (TN = 95.2%)      36  (FP = 4.8%)
ACTUAL SYNTHETIC (750) 23  (FN = 3.1%)       727 (TP = 96.9%)
```

---

## 🚀 Key Capabilities & Features

1. **Dual Input Methods:**
   * **Modern Drag-and-Drop Dropzone:** Dashed-blue upload area accepting JPG, JPEG, PNG, WebP, and BMP files up to 15 MB.
   * **Live On-Spot Webcam Capture:** Native `st.camera_input` tab allowing analysts or users to take a live photo and receive instantaneous veracity verification.
2. **Defensive File Safety Gate:** Strict 16-byte magic number validation, memory decompression bomb protection (`MAX_IMAGE_PIXELS = 16,777,216`), and filename traversal sanitization.
3. **Dual-Scale Multi-View Inference:** Simultaneous evaluation of the full portrait composition (resampling boundaries, background blending) and an 80% central face ROI zoom (biological skin pores, iris textures, corneal specular reflections).
4. **Asymmetric Forensic Fusion:** Prioritizes strong generative cues while preserving confirmed human biological micro-textures.
5. **Probability Calibration (Platt Temperature Scaling):** Scaled via empirical temperature $T = 0.9041$, ensuring output percentages reflect real-world statistical confidence.
6. **Dual Analysis & Decision Modes:**
   * **Direct Verdict Mode ($\tau = 0.58$):** Immediate binary classification ($96.07\%$ accuracy) optimized for low false alarms.
   * **Forensic 3-Tier Mode:** Safety buffer with explicit abstention ($\le 35\%$ Real, $\ge 65\%$ Fake, $35\%-65\%$ Inconclusive / Human Triage).
7. **Dual Audit Export:**
   * **Executive PDF Audit Certificate:** One-click download of a high-resolution, tamper-evident A4 PDF document containing image thumbnail, status badge, calibrated likelihood, raw logit, deep telemetry table, and legal notice.
   * **Machine-Readable JSON Record:** Complete diagnostic dump for programmatic logging or SIEM integration.

---

## 📂 Complete Directory & File Architecture

```
Deepfake Image Detection/
├── .venv/                         # Isolated Python 3.10.11 Virtual Environment (CUDA 12.4 Torch)
├── assets/
│   └── demo/                      # Curated test assets for instant sidebar testing
│       ├── real_01.jpg            # Authentic portrait sample 1
│       ├── real_02.jpg            # Authentic portrait sample 2
│       ├── fake_sd_01.jpg         # Stable Diffusion synthetic portrait 1
│       ├── fake_sd_02.jpg         # Stable Diffusion synthetic portrait 2
│       ├── fake_mj_01.jpg         # Midjourney synthetic portrait
│       ├── edge_lowres.jpg        # Low resolution (<128px) quality abstention sample
│       ├── edge_blur.jpg          # Blurred portrait sample (Laplacian < 100)
│       └── edge_corrupt.png       # Corrupted header security rejection sample
├── configs/
│   ├── preprocess.json            # Canonical preprocessing, temperature T=0.9041, thresholds
│   └── train_config.json          # Two-stage training hyperparameters, learning rates, seed
├── data/
│   ├── raw/
│   │   ├── new_real/              # 50,000 Authentic FFHQ portrait images
│   │   └── new_fake_stylegan/     # 50,000 StyleGAN2 synthetic portrait images
│   └── processed/                 # Standardized train, validation, and locked test images
├── docs/
│   ├── project_setup_guide.md     # Step-by-Step Installation & System Execution Guide
│   ├── architecture.md            # Comprehensive Software Architecture & Contracts (v2.0)
│   ├── model_training.md          # 10k Training Specification & Telemetry Guide
│   ├── tech_stack.md              # Technology Stack rationale, versions, and role breakdown
│   ├── datasets.md                # Dataset governance, pHash deduplication & ingestion
│   ├── developer_guide.md         # Operational developer runbook & recipes
│   ├── ethics_and_limitations.md  # Responsible AI, forensic limitations & legal guidelines
│   ├── prd.md                     # Product Requirements Document
│   └── tasks.md                   # Complete implementation and verification task roadmap
├── manifests/
│   └── manifest.json              # Cryptographic JSON catalog of 10,000 balanced images
├── models/
│   └── best_model.pth             # Trained PyTorch EfficientNet-B0 weights checkpoint
├── reports/
│   ├── metrics.json               # Full statistical metrics evaluated on 1,500 test images
│   ├── confusion_matrix.png       # High-resolution annotated confusion matrix
│   ├── roc_curve.png              # Receiver Operating Characteristic plot
│   └── calibration_curve.png      # Reliability diagram showing Platt calibration
├── src/
│   ├── security.py                # Magic bytes check, decompression bomb & path defense
│   ├── preprocess.py              # EXIF harvesting, orientation, RGB norm, Laplacian blur
│   ├── dataset.py                 # 64-bit DCT pHash deduplication, dataset manifests, DataLoader
│   ├── model.py                   # EfficientNet-B0 topology & specialized forensic head
│   ├── train.py                   # Two-stage fine-tuning engine with early stopping
│   ├── calibration.py             # L-BFGS Platt temperature scaling optimization
│   ├── predict.py                 # Multi-view inference, fusion, and decision engine
│   ├── evaluate.py                # Batch offline evaluation & report generation
│   └── report_pdf.py              # Vector/Raster Executive PDF Certificate Generator
├── tests/
│   ├── test_security.py           # Unit tests for magic bytes, limits, sanitization
│   ├── test_preprocess.py         # Unit tests for EXIF, orientation, resizing, Laplacian
│   ├── test_dataset.py            # Unit tests for pHash, deduplication, and splits
│   ├── test_model.py              # Unit tests for model forward pass & parameter freezing
│   └── test_inference.py          # Unit tests for inference, decision engine, and PDF export
├── app.py                         # Interactive Streamlit dashboard & CLI diagnostic runner
├── pytest.ini                     # PyTest test runner configuration
├── requirements.txt               # Complete Python package dependency manifest
└── README.md                      # This master reference document
```

---

## ⚡ Quickstart: Setup & Execution Guide

### 1. Environment & Prerequisites
* **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
* **Python Runtime:** Python 3.10.x or 3.11.x recommended
* **Compute:** NVIDIA GPU with CUDA 12.x (or CPU fallback mode)

### 2. Activate Virtual Environment
The project comes pre-configured with a virtual environment in `.venv`:
```powershell
# In PowerShell (Windows):
.\.venv\Scripts\Activate.ps1

# Verify GPU availability:
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. Launch the Interactive Web Dashboard
Run the Streamlit application:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open your web browser and navigate to: **`http://localhost:8501`**

### 4. Standalone CLI Diagnostic Mode
You can also run image inspections directly from the command line without launching a browser:
```powershell
python app.py --image assets/demo/real_01.jpg
```

### 5. Run Automated Test Suite
Verify that all unit tests, security gates, inference rules, and PDF generation pass 100%:
```powershell
pytest tests/ -v
```

---

## 🖥️ How to Use the Interactive Dashboard

1. **Select Input Mode:**
   * **📁 Drag & Drop / File Upload:** Drag any image from your computer and drop it directly onto the dashed-blue upload box (or click "Browse files").
   * **📸 Live Camera Capture:** Click the camera tab, grant browser webcam permissions, position your face in good lighting, and click **"Take Photo"**.
   * **Sidebar Demo Gallery:** Or choose one of the 8 curated demo assets from the sidebar dropdown.
2. **Select Decision Mode (Sidebar):**
   * **⚡ Direct Verdict Mode (Default):** Evaluates against the $\tau = 0.58$ threshold for immediate Real vs. Fake classification.
   * **🔬 Forensic 3-Tier Mode:** Evaluates against safety margins ($\le 35\%$ Real, $\ge 65\%$ Fake, $35\%-65\%$ Abstain) for legal or KYC triage.
3. **Inspect Diagnostic Telemetry & Verdict:**
   * **Status Badge:** Green (`Consistent with Natural Optical Capture`), Red (`AI-Generated / Synthetic Traces Detected`), or Amber (`Inconclusive / Quality Degraded`).
   * **Calibrated Likelihood:** Primary percentage gauge showing deepfake probability.
   * **Model Latency & SHA-256:** Hardware execution time and cryptographic hash.
   * **Deep Forensic Expander:** Detailed sensor telemetry (Laplacian focus variance, Shannon entropy, EXIF camera make/model/software).
4. **Download Forensic Reports:**
   * Click **`📄 Download Audit Report (PDF)`** to download the official forensic audit certificate.
   * Click **`📥 Download Audit Record (JSON)`** to download the raw telemetry dictionary.

---

## 🔍 Complete Dashboard Metrics & Telemetry Guide

| Metric Name | Display Location | Mathematical Definition / Meaning | Normal / Expected Range | Forensic Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Calibrated Synthetic Likelihood** | Main Verdict Card | $s = \sigma\left(\frac{z}{0.9041}\right) \times 100\%$ | $0.0\% - 100.0\%$ | Estimated probability of AI generation. E.g. $5.2\%$ means **$94.8\%$ authentic confidence**. |
| **Raw Logit ($z$)** | Below Likelihood | Linear layer output before sigmoid scaling | $(-\infty, +\infty)$ | **Negative ($< 0$):** Real optical capture.<br>**Positive ($> 0$):** AI generative traces. |
| **Model Latency** | Middle Telemetry | GPU forward pass time for dual-scale views | $120\text{ ms} - 180\text{ ms}$ (RTX 3050) | Confirms real-time processing performance. |
| **Total Pipeline** | Middle Telemetry | End-to-end time (ingestion + preprocessing + GPU + UI) | $180\text{ ms} - 240\text{ ms}$ | Complete wall-clock request turnaround. |
| **SHA-256 Digest** | Middle Telemetry | 256-bit cryptographic hash of raw image bytes | 64 Hex Characters | Immutable digital fingerprint proving image integrity (Chain of Custody). |
| **Focus Laplacian Variance** | Deep Expander | $\text{Var}\left( \nabla^2 I \right)$ | $\ge 100.0$ (Sharp)<br>$< 100.0$ (Soft / Blurry) | Measures edge sharpness and skin pore clarity. Values $< 100$ trigger a blur advisory. |
| **Shannon Entropy** | Deep Expander | $-\sum_{k=0}^{255} p(k) \log_2 p(k)$ | $7.0 - 7.8\text{ bits/pixel}$ | Measures information richness and texture randomness across color channels. |
| **Camera Make / Model** | Deep Expander | EXIF metadata tags from camera hardware | E.g. Sony, Apple, Canon, or `None` | Web platforms (WhatsApp, Twitter) strip EXIF for privacy, so `None` is standard. |
| **Software Flag** | Deep Expander | EXIF software / editing application tag | E.g. "Photoshop", "Inpaint", or `None` | Flags image manipulation software if present. |
| **Temperature $T$** | Sidebar Telemetry | Platt scaling optimization factor | $T = 0.9041$ | Calibrates neural logits to empirical probabilities ($ECE = 0.0118$). |
| **Decision Rule** | Sidebar Telemetry | Threshold applied to calibrated score | Direct: $\tau = 0.58$<br>3-Tier: $\le 0.35 / \ge 0.65$ | Governs whether score triggers Verified, Flagged, or Inconclusive. |

---

## 🏗️ System Architecture & Processing Pipeline

```
[User Image (Drag & Drop) OR Live Webcam Photo]
                    │
                    ▼
[Security & File Safety Gate (src/security.py)]
  ├── 16-Byte Magic Bytes Validator (JPEG / PNG / WebP)
  ├── Decompression Bomb Protection (Area ≤ 16.7M pixels)
  └── Path Traversal Sanitizer & SHA-256 Hash Computation
                    │ (Passes Ingestion)
                    ▼
[Forensic Preprocessing & Quality (src/preprocess.py)]
  ├── EXIF Orientation Transposer & 3-Channel RGB Standardizer
  ├── Focus Laplacian Sharpness Variance & Shannon Entropy Calculation
  └── Dual-Scale Multi-View Construction:
        ├─ View 1: Global Composition View (224x224 px)
        └─ View 2: Central Face ROI Zoom (80% crop -> 224x224 px)
                    │
                    ▼
[Neural Inference Engine (src/predict.py & src/model.py)]
  ├── PyTorch EfficientNet-B0 Forward Passes on CUDA (RTX 3050)
  └── Asymmetric Forensic Fusion:
        • If View 1 logit ≥ 1.0 ──► Use View 1 (Global generative boundary trace)
        • If View 2 logit ≤ -1.2 ──► Use View 2 (Natural biological pores confirmed)
        • Otherwise ──────────────► Balanced 50/50 combination
                    │
                    ▼
[Platt Calibration & Decision Engine]
  ├── Temperature Scaling: s = σ(z / 0.9041)
  └── Decision Mode Evaluation (Direct Mode τ=0.58 OR Forensic 3-Tier Mode)
                    │
                    ▼
[Presentation & Certification Layer (app.py & src/report_pdf.py)]
  ├── Interactive Visual Inspection & Telemetry Gauge
  ├── 1-Click Forensic Audit Record (JSON)
  └── 1-Click Executive Forensic PDF Audit Certificate (A4 150 DPI)
```

---

## 🔄 Offline Model Training & Retraining Protocol

If you ever wish to retrain or fine-tune the model with new data in the future:

### 1. Ingest Images:
Place your authentic images into `data/raw/new_real/` and synthetic images into `data/raw/new_fake_stylegan/`.

### 2. Generate Balanced Manifest:
```powershell
python -c "
from src.dataset import build_dataset_manifest
manifest = build_dataset_manifest(
    real_dir='data/raw/new_real',
    fake_dir='data/raw/new_fake_stylegan',
    output_manifest_path='manifests/manifest.json',
    max_samples_per_class=5000,
    seed=42
)
print('Manifest generated with', len(manifest), 'samples.')
"
```

### 3. Run Two-Stage GPU Training:
```powershell
python src/train.py
```
This executes Stage 1 (Head warmup, 3 epochs) and Stage 2 (Fine-tuning, 18 epochs) on your NVIDIA RTX 3050 GPU, saving the peak validation model to `models/best_model.pth`.

### 4. Optimize Temperature Calibration & Evaluate Test Set:
```powershell
python src/evaluate.py
```
This fits temperature $T$ using validation logits, evaluates the 1,500 held-out test split, updates `reports/metrics.json`, and generates high-resolution confusion matrix and ROC curve plots in `reports/`.

---

## 🛠️ Operational Troubleshooting & Common Scenarios

### Scenario 1: Identical Numeric Filenames in Kaggle Dataset
* **Issue:** Kaggle datasets often number both real and fake images with identical names (e.g. `00091.jpg` in `real/` and `00091.jpg` in `fake/`).
* **Solution:** Always verify whether the image was copied from the `real/` folder or the `fake/` folder. Checking the SHA-256 hash or testing the sample confirmed in `data/raw/new_real/` will verify authenticity.

### Scenario 2: Live Webcam Photo Flagged as Blurry
* **Issue:** Webcam images in dim lighting can produce a "Focus / Blur Detected" advisory banner (`Laplacian Variance < 100.0`).
* **Solution:** Ensure adequate face lighting and keep the camera steady. Natural skin pores require sharp focus to confirm biological textures.

### Scenario 3: Missing Camera Metadata ("None / None")
* **Issue:** Dashboard displays `Camera Make / Model: None / None`.
* **Explanation:** This is standard for web, social media, and messaging transfers (WhatsApp, Twitter) which strip EXIF data. The model does **not** treat missing EXIF as a sign of deepfake; it evaluates raw pixel patterns.

### Scenario 4: Switching Between Direct Mode & 3-Tier Mode
* **Direct Mode:** Best for testing single images where you want a decisive "Real vs. Fake" verdict ($96.07\%$ accuracy).
* **Forensic 3-Tier Mode:** Best for regulatory/KYC environments where borderline images ($35\%-65\%$) should be routed to human compliance officers rather than auto-rejected.

---

## ⚖️ Legal, Ethical & Forensic Disclaimers

1. **Academic & Forensic Triage Prototype:** This software is designed to assist human analysts in detecting synthetic image anomalies. It does **not** constitute standalone, indisputable legal proof of fraud or authenticity in a court of law.
2. **Non-Biometric Assessment:** The system does **not** perform 1:1 facial matching or 1:N facial recognition, and does not assess physical human liveness (e.g., blink detection or 3D depth).
3. **Continuous Arms Race:** Generative AI models (Diffusion, GANs) continuously evolve. Regular retraining with diverse multi-generator datasets is necessary to maintain high detection efficacy.
4. **Mandatory Human-in-the-Loop:** All flagged or inconclusive determinations in high-stakes environments (banking, KYC, passport auditing) must be reviewed by a certified forensic specialist.

---
*Maintained as part of the AI-Based Deepfake Image Detection Project • Academic & Forensic Prototype v2.0.0*
