# 🚀 Complete Project Setup & Execution Guide

## AI-Based Deepfake Image Detection System for Digital Identity Verification
**Document Version:** 1.0.0 (End-to-End Installation & Deployment Runbook)  
**Associated Project Master:** [`README.md`](../README.md)  
**Associated Architecture:** [`docs/architecture.md`](./architecture.md)  
**Associated Model Training:** [`docs/model_training.md`](./model_training.md)  
**Associated Tech Stack:** [`docs/tech_stack.md`](./tech_stack.md)  

---

## 1. System Requirements & Prerequisites

Before setting up the project, ensure your workstation meets the following minimum requirements:

### Hardware Requirements
* **Operating System:** Windows 10/11 (64-bit), Ubuntu Linux 20.04/22.04 LTS, or macOS 12+
* **Processor (CPU):** Intel Core i5/i7 (8th Gen or newer) or AMD Ryzen 5/7
* **System RAM:** 8 GB minimum (16 GB recommended)
* **Graphics (GPU):**
  * **Recommended:** NVIDIA GPU with CUDA 12.x support (e.g. GeForce RTX 3050, RTX 3060, RTX 4060, or better with $\ge 4\text{ GB}$ VRAM).
  * **Fallback:** CPU execution is fully supported automatically if an NVIDIA GPU is not detected.
* **Disk Space:** At least 2.5 GB free space for virtual environment, PyTorch wheels, and model weights.

### Software Prerequisites
* **Python Runtime:** Python **3.10.x** or **3.11.x** (Python 3.12+ is supported, but 3.10/3.11 provides maximum stability for prebuilt PyTorch CUDA wheels).
  * Verify your installed version:
    ```bash
    python --version
    ```
* **Web Browser:** Google Chrome, Microsoft Edge, Mozilla Firefox, or Brave (needed for the Streamlit dashboard and webcam capture).

---

## 2. Step-by-Step Setup Instructions

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SETUP PROCESS AT A GLANCE                       │
│                                                                        │
│  [1. Navigate to Project] ──► [2. Create Virtual Environment (.venv)]  │
│                                            │                           │
│                                            ▼                           │
│  [4. Install Dependencies] ◄── [3. Activate Virtual Environment]       │
│             │                                                          │
│             ▼                                                          │
│  [5. Verify GPU & Checkpoint] ──► [6. Launch Streamlit Web Dashboard]  │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Step 1: Open Terminal & Navigate to Project Directory

Open your command line interface (**PowerShell** on Windows, or **Bash/Zsh** on Linux/macOS) and navigate into the project root directory:

```powershell
# Windows PowerShell / CMD:
cd "c:\Users\loq\Downloads\Deepfake Image Detection"

# Linux / macOS:
cd ~/Downloads/"Deepfake Image Detection"
```

---

### Step 2: Create an Isolated Virtual Environment

It is strongly recommended to use a dedicated virtual environment (`.venv`) so the project dependencies do not interfere with other Python installations on your computer:

```powershell
# Create a virtual environment named .venv
python -m venv .venv
```

*(Note: If you have multiple Python versions installed on Windows, specify Python 3.10 explicitly, e.g.: `py -3.10 -m venv .venv`)*

---

### Step 3: Activate the Virtual Environment

Activate the environment depending on your operating system:

#### On Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```
> **⚠️ If you see an error saying `script execution is disabled on this system`:**  
> Run this one-time command in PowerShell to allow virtual environment activation:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```

#### On Windows (Command Prompt - CMD):
```cmd
.\.venv\Scripts\activate.bat
```

#### On Linux / macOS (Bash / Zsh):
```bash
source .venv/bin/activate
```

*When successfully activated, your terminal prompt will show `(.venv)` at the beginning.*

---

### Step 4: Upgrade Pip & Install Dependencies

#### Option A: Recommended Installation for NVIDIA GPUs (CUDA 12.4):
If your machine has an NVIDIA GeForce/RTX graphics card:
```powershell
# 1. Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# 2. Install PyTorch with CUDA 12.4 acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install remaining application dependencies
pip install -r requirements.txt
```

#### Option B: Standard Installation (CPU-only / Non-NVIDIA / Mac):
If you do not have an NVIDIA GPU or are running on macOS:
```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

### Step 5: Verify Environment & Hardware Acceleration

Run this quick one-line diagnostic command to confirm your setup:

```powershell
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Running on CPU')"
```

**Expected Output (NVIDIA GPU System):**
```
PyTorch Version: 2.6.0+cu124
CUDA Available: True
Device Name: NVIDIA GeForce RTX 3050 Laptop GPU
```

---

### Step 6: Verify Model Checkpoint

Ensure the trained model weights file is present in the `models/` directory:
```powershell
# In PowerShell:
Test-Path models/best_model.pth

# In Linux / macOS:
ls -lh models/best_model.pth
```
*Expected: File exists and is $\approx 18\text{ MB}$ in size.*

---

## 3. Running the Application

### 3.1 Launching the Interactive Web Dashboard

To start the local web application:

```powershell
streamlit run app.py
```
*(Or alternatively: `python -m streamlit run app.py`)*

After launching, your terminal will display:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

1. Open your browser and go to **`http://localhost:8501`**.
2. You will see the **AI-Based Deepfake Image Detection System** dashboard.

---

### 3.2 Running the Standalone CLI Tool (No Browser Needed)

To inspect an image directly from the command line:

```powershell
python app.py --image assets/demo/real_01.jpg
```

**Sample CLI Output:**
```
======================================================================
AI-BASED DEEPFAKE IMAGE DETECTION SYSTEM — FORENSIC DIAGNOSTIC SUITE
======================================================================
Model Backbone: efficientnet_b0 on CUDA
Calibration Temperature T: 0.9041
Analyzing Target Image: assets/demo/real_01.jpg
----------------------------------------------------------------------
VERDICT:           [VERIFIED] Consistent with Natural Optical Capture
CALIBRATED SCORE:  5.2% Synthetic Probability
RAW LOGIT:         -4.5321
IMAGE DIMENSIONS:  600 x 600 px (JPEG)
SHA-256 HASH:      aa1609c73bfa...
LAPLACIAN FOCUS:   245.26 (Threshold: >= 100.0)
MODEL LATENCY:     142.1 ms (Total Pipeline: 190.3 ms)
----------------------------------------------------------------------
LEGAL NOTICE: Diagnostic triage tool only. Does not prove liveness or fraud.
======================================================================
```

---

### 3.3 Running Automated Unit & Integration Tests

To run all automated verification tests:

```powershell
pytest tests/ -v
```

**Expected Output:**
```
tests/test_dataset.py::test_build_dataset_manifest PASSED
tests/test_evaluate.py::test_compute_comprehensive_metrics PASSED
tests/test_inference.py::test_evaluate_decision_thresholds PASSED
tests/test_inference.py::test_evaluate_decision_direct_mode PASSED
tests/test_inference.py::test_pdf_report_generation PASSED
tests/test_model.py::test_build_model_efficientnet PASSED
tests/test_preprocess.py::test_resize_image_deterministic PASSED
tests/test_security.py::test_validate_magic_bytes PASSED
============================== 29 passed in 30.12s ==============================
```

---

## 4. How to Use the Dashboard Features

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DASHBOARD WORKFLOW                              │
├────────────────────────────────────────────────────────────────────────┤
│ 1. CHOOSE INPUT:                                                       │
│    • Tab 1: Drag & drop any JPG/PNG image into the blue dropzone       │
│    • Tab 2: Click 'Live Camera Capture' & take a photo with webcam     │
│    • Sidebar: Select a curated sample from 'Select Demo Gallery Asset' │
│                                                                        │
│ 2. CHOOSE DECISION MODE (Sidebar):                                     │
│    • ⚡ Direct Verdict Mode (Default, 96.07% accuracy, tau=0.58)       │
│    • 🔬 Forensic 3-Tier Mode (35% Real / 65% Fake / Middle Abstain)    │
│                                                                        │
│ 3. INSPECT DIAGNOSTICS:                                                │
│    • Left: Image Preview & Dimensions                                  │
│    • Right: Status Badge (Green/Red/Amber), Likelihood Gauge, Latency  │
│    • Expander: Laplacian Focus Sharpness, Shannon Entropy, EXIF Tags   │
│                                                                        │
│ 4. EXPORT AUDIT REPORTS:                                               │
│    • 📄 Download Audit Report (PDF): Official tamper-evident PDF       │
│    • 📥 Download Audit Record (JSON): Raw machine-readable logs        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Operational Troubleshooting & FAQs

### Q1: When running `Activate.ps1`, PowerShell says `Execution of scripts is disabled`.
* **Fix:** Open PowerShell as your normal user and execute:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
  Then run `.\.venv\Scripts\Activate.ps1` again.

### Q2: When I click on "Live Camera Capture", my webcam doesn't open.
* **Fix:** 
  1. Check your browser address bar (top left near `localhost:8501`) and ensure Camera permissions are set to **"Allow"**.
  2. Ensure another application (Zoom, Teams, Skype, or Windows Camera app) is not actively locking your webcam.

### Q3: Why does Streamlit say `Port 8501 is already in use`?
* **Fix:** If another instance is running in the background, you can either stop it or specify an alternative port:
  ```powershell
  streamlit run app.py --server.port 8502
  ```

### Q4: I tested a photo and it shows `Camera Make / Model: None / None`. Is this an error?
* **Explanation:** No, this is completely normal. Web browsers, social media sites (Twitter, Instagram), and messaging apps (WhatsApp, Telegram) automatically strip EXIF metadata from images to protect user privacy. The AI model evaluates raw pixel micro-textures, not EXIF tags.

### Q5: Can I run this system on a machine without a dedicated GPU?
* **Yes!** The system automatically detects whether CUDA is available:
  * If an NVIDIA GPU with CUDA is present, it uses GPU acceleration ($\approx 140\text{ ms}$ inference).
  * If no GPU is present, it automatically runs in CPU mode ($\approx 400 - 600\text{ ms}$ inference). No manual code changes are required.

---

## 6. How to Stop the Application

To shut down the Streamlit server:
1. In the terminal window running Streamlit, press **`Ctrl + C`**.
2. To deactivate the Python virtual environment, type:
   ```powershell
   deactivate
   ```

---
*Maintained as part of the AI-Based Deepfake Image Detection Project • Project Setup & Execution Guide v1.0.0*
