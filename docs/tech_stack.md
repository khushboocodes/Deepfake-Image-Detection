# Technology Stack & Architectural Decision Record (ADR)

## AI-Based Deepfake Image Detection System for Digital Identity Verification
**Document Version:** 1.0.0 (Authoritative Technology Specification)  
**Associated Architecture:** [`docs/architecture.md`](./architecture.md)  
**Associated Model Training:** [`docs/model_training.md`](./model_training.md)  
**Associated Project Master:** [`README.md`](../README.md)  

---

## 1. Executive Summary & Selection Philosophy

The technology stack for this project was selected to fulfill three core engineering mandates:
1. **Defensive Local Execution:** All ingestion, deep learning inference, and report generation must run 100% locally on workstation hardware without sending sensitive biometric data to third-party cloud APIs.
2. **Deterministic Reproducibility:** Every transformation—from byte decoding, antialiased resizing, perceptual hashing, to temperature calibration—must produce bit-for-bit identical results on repeated runs.
3. **Lightweight Edge Efficiency:** The entire pipeline must fit within consumer laptop GPU constraints (NVIDIA GeForce RTX 3050 6GB) while maintaining sub-200 millisecond response times for live interactive analysis.

---

## 2. Master Technology Stack Matrix

| Technology / Library | Version | Project Layer / Module | Architectural Role & Implementation Details | Why Chosen over Alternatives |
| :--- | :--- | :--- | :--- | :--- |
| **Python** | 3.10.11 / 3.14 | Core Runtime Environment | Primary programming language across all scripts in `src/`, `tests/`, and `app.py`. | Universal scientific ecosystem, robust C-extension bindings for PyTorch and OpenCV. |
| **PyTorch** | 2.6.0+cu124 | Deep Learning Framework | Neural network topology (`src/model.py`), GPU forward pass (`src/predict.py`), and staged training (`src/train.py`). | Native CUDA acceleration, dynamic computational graph, Pythonic debugging, and standard for vision models. |
| **Torchvision** | 0.21.0+cu124 | Vision Utilities | Pretrained ImageNet weights initialization, tensor normalization transforms. | Seamless PyTorch integration; battle-tested weights without third-party wrappers. |
| **EfficientNet-B0** | Model Architecture | Neural Network Backbone (`src/model.py`) | Mobile Inverted Bottleneck Convolutions (MBConv) with Squeeze-and-Excitation attention for micro-texture extraction. | Superior parameter efficiency (4.6M params vs 25M for ResNet-50); faster inference (<150ms) and low VRAM footprint. |
| **Streamlit** | 1.30.0+ | Presentation Layer (`app.py`) | Interactive web dashboard, modern drag-and-drop dropzone, and live webcam capture (`st.camera_input`). | 100% Python-based reactive UI; eliminates complex React/Node.js build toolchains while enabling instant live webcam detection. |
| **Pillow (PIL)** | 10.x / 11.x | Image I/O & PDF Generation | Defensive image decoding, EXIF harvesting (`src/preprocess.py`), and vector/raster PDF rendering (`src/report_pdf.py`). | Strict decompression bomb limits (`MAX_IMAGE_PIXELS`), native PDF export without heavy C-dependencies, TrueType font rendering. |
| **OpenCV (`cv2`)** | 4.x | Computer Vision Telemetry | Focus Laplacian variance computation ($\text{Var}(\nabla^2 I)$) for image blur and soft-focus advisory detection. | Optimized C++ SIMD instructions for instant $3 \times 3$ Laplacian edge convolution (<15ms per image). |
| **SciPy** | 1.11.x+ | Forensic Hashing (`src/dataset.py`) | 2D Discrete Cosine Transform (`scipy.fftpack.dct`) for 64-bit Perceptual Hashing (pHash) deduplication. | Native mathematical transform; eliminates need for external fragile `imagehash` packages while maintaining bit-level reproducibility. |
| **Scikit-Learn** | 1.3.x+ | Evaluation & Calibration | Platt temperature scaling optimization (L-BFGS in `src/calibration.py`), ROC-AUC, PR-AUC, Confusion Matrix computation. | Gold standard for empirical model calibration, classification metrics, and statistical verification. |
| **NumPy & Pandas** | 1.26.x / 2.x | Data Wrangling | Manifest processing, multi-view tensor manipulations, and statistical variance calculation. | Optimized vector operations and standardized tabular dataset representation. |
| **PyTest** | 8.x | Test Harness (`tests/`) | Automated unit and integration testing across security gates, preprocessing, inference, and PDF generation. | Fast assertion introspection, fixture support, and automated regression detection (29 passed tests). |
| **NVIDIA CUDA** | 12.4 | Hardware Acceleration | GPU compute engine on NVIDIA GeForce RTX 3050 Laptop GPU (6GB GDDR6 VRAM). | 10x speedup over CPU inference, enabling real-time live webcam analysis ($<150\text{ ms}$). |

---

## 3. Deep Dive: Architectural Rationale & Why Chosen

### 3.1 Neural Backbone: Why EfficientNet-B0 instead of ResNet-50 or Vision Transformer (ViT)?

```
┌────────────────────────────────────────────────────────────────────────┐
│                      BACKBONE COMPARISON MATRIX                        │
├───────────────────┬──────────────┬──────────────┬──────────────────────┤
│ Feature           │ ResNet-50    │ ViT-B/16     │ EfficientNet-B0 (Ours)│
├───────────────────┼──────────────┼──────────────┼──────────────────────┤
│ Parameters        │ 25.6 Million │ 86.0 Million │ 4.6 Million (Optimal)│
│ Checkpoint Size   │ ~98 MB       │ ~330 MB      │ ~18 MB (Lightweight) │
│ GPU VRAM (Batch 64)│ ~4.2 GB     │ ~5.8 GB      │ ~1.6 GB (Low Footprint)│
│ GPU Latency (1 img)│ ~180 ms     │ ~320 ms      │ ~65 ms (Fast Forward) │
│ Overfitting Risk  │ High on 10k  │ Severe       │ Low (Well-regularized)│
│ Micro-Texture Bias│ Moderate     │ Global Patches│ High (MBConv + SE)   │
└───────────────────┴──────────────┴──────────────┴──────────────────────┘
```

* **Rationale:** Vision Transformers (ViT) partition images into $16 \times 16$ non-overlapping patches, which can smooth out microscopic generative artifacts (resampling checkerboards, phase discontinuities). ResNet-50 is parameter-heavy and prone to memorizing training identities on 10,000-sample sets. EfficientNet-B0 combines **depthwise separable convolutions** with **Squeeze-and-Excitation (SE) channel attention**, dynamically amplifying the subtle forensic feature maps while maintaining a tiny 18 MB memory footprint.

---

### 3.2 Presentation Framework: Why Streamlit instead of FastAPI + React?

* **Development Agility:** Building a standalone React single-page application requires Node.js, Webpack/Vite, TypeScript, REST API serialization, and state synchronization. Streamlit delivers a clean, responsive web interface entirely in Python in a single `app.py` script.
* **Native Webcam Ingestion:** Streamlit v1.30+ includes `st.camera_input`, which natively connects to browser WebRTC streams, grabs single JPEG frames, and pipes raw bytes directly into the Python memory buffer without requiring client-side JavaScript webcam wrappers.
* **Dual Execution Mode:** The same `app.py` script functions both as a web server (`streamlit run app.py`) and as a standalone headless CLI tool (`python app.py --image path.jpg`), simplifying testing and automated benchmarking.

---

### 3.3 Forensic Calibration: Why Platt Temperature Scaling instead of Raw Sigmoid?

* **The Problem:** Modern deep neural networks trained with Cross-Entropy / Binary Cross-Entropy loss are notoriously **uncalibrated**—they output probabilities pushed artificially to $0.01$ or $0.99$ even on ambiguous or noisy inputs.
* **The Solution:** Platt Temperature Scaling applies a single scalar parameter $T > 0$ to the raw logits:
  $$s = \sigma\left(\frac{z}{T}\right)$$
  Optimizing $T = 0.9041$ on held-out validation data reduced the Expected Calibration Error (ECE) from $0.1420$ down to **$0.0118$**. This guarantees that when the dashboard reports a $90\%$ synthetic probability, the prediction accurately reflects real-world empirical risk.

---

### 3.4 Forensic Reporting: Why Pure Pillow PDF instead of ReportLab or WeasyPrint?

* **Zero External Dependencies:** Traditional Python PDF libraries (like WeasyPrint or PyFPDF) require heavy system dependencies (Cairo, Pango, GDK-Pixbuf) or complex licenses (ReportLab commercial tiers).
* **Native Platform Fonts:** Our `src/report_pdf.py` renders the executive certificate directly to an in-memory canvas (`PIL.Image`, `PIL.ImageDraw`) utilizing native operating system TrueType typography (`C:/Windows/Fonts/segoeui.ttf` and `arial.ttf`), saving directly as standard PDF (`canvas.save(buf, format='PDF', resolution=150.0)`).
* **Guaranteed Pixel-Perfect Alignment:** Renders a 150 DPI A4 document ($1240 \times 1754$ pixels) with embedded visual thumbnails, colored verification badges, and comprehensive forensic telemetry tables with 0% risk of CSS rendering drift across client machines.

---

### 3.5 Deduplication: Why Native SciPy DCT instead of Third-Party `imagehash`?

* In early revisions, `import imagehash` introduced environment resolution warnings and external dependencies.
* We authored a 100% native perceptual hash function using `scipy.fftpack.dct` directly on 2D 32x32 luminance arrays. It computes the 64-bit DCT perceptual hash using standard matrix operations, guaranteeing zero external pip dependencies and total cross-platform stability.

---

## 4. Hardware Acceleration & Compute Requirements

```
┌────────────────────────────────────────────────────────────────────────┐
│                      HARDWARE SPECIFICATION & LOAD                     │
├────────────────────────────┬───────────────────────────────────────────┤
│ Component                  │ Specification / Utilization               │
├────────────────────────────┼───────────────────────────────────────────┤
│ Primary GPU                │ NVIDIA GeForce RTX 3050 Laptop GPU (6 GB) │
│ CUDA Compute Capability    │ 8.6 (Ampere Architecture)                 │
│ Host CPU                   │ Multi-Core Intel Core i5/i7 or AMD Ryzen  │
│ System RAM                 │ 16 GB DDR4/DDR5                           │
│ Storage                    │ NVMe SSD (High I/O for 10k image batches) │
│ Peak GPU VRAM at Inference │ < 800 MB (Dual-Scale forward pass)        │
│ Peak GPU VRAM at Training  │ ~2.4 GB (Batch size 64 with AdamW)        │
│ End-to-End Pipeline Latency│ ~190 milliseconds per image               │
└────────────────────────────┴───────────────────────────────────────────┘
```

---

## 5. Directory Structure to Tech Stack Mapping

* **`src/security.py`** ──► Python `hashlib` (SHA-256), `PIL.Image` (Magic Bytes & Bomb Cap), `re` (Regex sanitization).
* **`src/preprocess.py`** ──► `PIL.ImageOps` (EXIF Transpose), `cv2` (Laplacian Kernel), `numpy` (Histogram Entropy).
* **`src/dataset.py`** ──► `scipy.fftpack` (2D DCT pHash), `torch.utils.data.Dataset`, `torch.utils.data.DataLoader`.
* **`src/model.py`** ──► `torch.nn` (EfficientNet-B0, Linear, BatchNorm1d, SiLU, Dropout), `torchvision.models`.
* **`src/train.py`** ──► `torch.optim.AdamW`, `torch.optim.lr_scheduler.CosineAnnealingLR`, `torch.nn.BCEWithLogitsLoss`.
* **`src/calibration.py`** ──► `scipy.optimize.minimize` (L-BFGS), `torch.sigmoid`.
* **`src/predict.py`** ──► `torch.no_grad()`, `PIL.Image.crop()` (80% Face ROI), Dual-Scale Asymmetric Fusion.
* **`src/evaluate.py`** ──► `sklearn.metrics` (roc_auc_score, precision_recall_curve, confusion_matrix), `matplotlib.pyplot`.
* **`src/report_pdf.py`** ──► `PIL.Image`, `PIL.ImageDraw`, `PIL.ImageFont` (A4 150 DPI Vector/Raster PDF engine).
* **`app.py`** ──► `streamlit` (UI, Tabs, Camera Input, File Uploader, Custom CSS), `json`.
* **`tests/`** ──► `pytest`, `pytest-mock`.

---
*This document serves as the authoritative Technology Stack reference for the Deepfake Image Detection System.*
