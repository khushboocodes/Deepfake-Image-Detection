# Product Requirements Document (PRD)

## AI-Based Deepfake Image Detection System for Digital Identity Verification
**Document Version:** 2.0.0 (Comprehensive Forensic & Engineering Specification)  
**Target Delivery Window:** 3-Day Submission MVP & Production Roadmap  
**System Classification:** Academic Machine Learning Prototype / Digital Forensics Diagnostic Tool  
**Operational Working Directory:** `c:\Users\loq\Downloads\Deepfake Image Detection`

---

## 1. Executive Summary & Vision Statement

### 1.1 Problem Statement
The proliferation of deep generative models—specifically Latent Diffusion Models (Stable Diffusion, Midjourney, DALL-E 3, Flux), Generative Adversarial Networks (StyleGAN2, StyleGAN3), and identity-swapping architectures (RoOP, SimSwap, InsightFace)—has drastically reduced the technical barrier to synthesizing photorealistic imagery. In digital identity verification (Know Your Customer / KYC, Anti-Money Laundering / AML, remote onboarding, credential auditing), synthetic and manipulated portrait images pose a severe systemic vulnerability. 

Traditional image forensics relying on naive metadata inspection or human inspection fail against modern generative pipelines. However, typical "black-box" deep learning classifiers often suffer from severe limitations:
1. **Catastrophic generalization failure**: Classifiers trained on one generator family (e.g., ProGAN) degrade to near-random performance when evaluated on unseen diffusion architectures.
2. **Shortcut learning**: Models exploit high-frequency JPEG grid artifacts, background watermarks, or dataset-specific resizing pipelines rather than genuine generative traces.
3. **Overconfident miscalibration**: Standard Softmax layers yield pseudo-probabilities ($>0.99$) on out-of-distribution (OOD) or degraded inputs, leading to dangerous false positives in identity workflows.
4. **Unsubstantiated forensic claims**: Misrepresenting a still-image classification score as "proof of identity", "proof of liveness", or "conclusive fraud evidence".

### 1.2 Core Value Proposition & System Mission
The **AI-Based Deepfake Image Detection System** is designed to provide an open, defensible, academically rigorous, and reproducible still-image forensic classification pipeline. The primary mission for this delivery is to build and validate a functional, deterministic end-to-end prototype that:
- Ingests arbitrary user-supplied still images (JPG/PNG).
- Enforces strict security boundaries against malformed or malicious inputs.
- Executes deterministic image sanitization, EXIF forensic logging, and standardized preprocessing.
- Runs inference via a fine-tuned deep convolutional transfer-learning baseline.
- Surfaces calibrated uncertainty, explicit abstention states ("Inconclusive"), and detailed forensic caveats.
- Provides a locked, reproducible evaluation harness with complete scientific metrics and error analysis.

### 1.3 Core Promise & Operational Pipeline
```
[User Image Upload] 
         │
         ▼
[Security & File Safety Gate] ──(Fails validation)──► [Abstention: Malformed / Rejected]
         │ (Passes: Valid Magic Bytes, Dimensions, Sanitized Headers)
         ▼
[Forensic Metadata Inspector] ──► Logs EXIF / ICC / Compression profile (Non-blocking evidence)
         │
         ▼
[Deterministic Preprocessing] ──► EXIF orientation normalize ➔ RGB ensure ➔ Bilinear resize (224x224) ➔ ImageNet Normalization
         │
         ▼
[Deep Learning Inference] ────► PyTorch Checkpoint (EfficientNet-B0 / ResNet-18) ➔ Logits
         │
         ▼
[Calibration & Abstention] ───► Temperature Scaling ➔ Decision Engine:
         │                           ├─ Score > 0.65 ➔ Flagged: AI-Generated / Synthetic
         │                           ├─ Score < 0.35 ➔ Verified: Real-Image-Consistent
         │                           └─ 0.35 ≤ Score ≤ 0.65 OR High Uncertainty ➔ Abstention: Inconclusive
         ▼
[Forensic Presentation UI] ───► Interactive Streamlit / Dashboard with Raw Preview, Confidence Metrics, 
                                 Metadata Breakdown, Limitations Caveat, and Audit Logging
```

### 1.4 Scope Boundaries: Academic MVP vs. Enterprise Roadmap

| Operational Dimension | Academic 3-Day Submission MVP (In Scope) | Enterprise Production Target (Out of Scope / Deferred) |
| :--- | :--- | :--- |
| **Primary Task** | AI-generated vs. Real still-image binary detection (GenImage benchmark subset) | Multi-modal forensic suite, Video temporal analysis, Audio-visual sync |
| **Input Modalities** | Static single image: JPEG, PNG format | Video streams (MP4/WebM), PDF ID scans, Live camera RTSP feeds |
| **Biometric Scope** | Non-biometric image consistency classification | Facial Recognition (1:1 / 1:N), Liveness / Presentation Attack Detection (PAD) |
| **Document KYC** | Basic photo integrity check only | Document OCR, MRZ parsing, Hologram validation, Barcode decryption |
| **Model Topology** | Single lightweight pretrained CNN backbone (EfficientNet-B0 / ResNet-18) | Multi-model ensemble (Frequency-domain DCT/FFT branch + Spatial CNN + ViT) |
| **Explainability** | Diagnostic forensic metrics, metadata extraction, optional Grad-CAM visualizer | Certified legal chain-of-custody reports, C2PA cryptographic provenance |
| **Deployment** | Local Python runtime, Streamlit UI, local PyTorch checkpoint | Multi-tenant Kubernetes cluster, SSO/OAuth2, Vault HSM, Distributed Celery workers |
| **Data Storage** | Ephemeral in-memory processing or local session temp files; zero external logging | Encrypted PostgreSQL/S3 audit vault, GDPR Right-to-Erasure compliance automation |

---

## 2. Stakeholder Profiles & Personas

### 2.1 Academic Evaluator / Thesis Reviewer
- **Objective:** Evaluate scientific rigor, methodological integrity, code reproducibility, proper metric reporting, and transparency regarding failure modes.
- **Key Needs:** Reproducible train/val/test splits without leakage; locked test evaluation script; confusion matrix, ROC-AUC, PR-AUC, and F1-score tables; honest discussion of limitations; adherence to open science principles.
- **Pain Points:** Hidden hyperparameters, cherry-picked demos, claiming 99.9% accuracy without out-of-distribution evaluation, mixing synthetic generator types across splits.

### 2.2 Digital Identity / KYC Risk Analyst (Human-in-the-Loop)
- **Objective:** Triage flagged applicant profile pictures or identity card portrait photos to prevent account takeover or synthetic identity fraud.
- **Key Needs:** A clear decision recommendation ("Real", "Synthetic", or "Inconclusive"), an intuitive visual score with calibrated uncertainty, and clear warnings that a single model score does not constitute legal proof of fraud.
- **Pain Points:** Forced binary outputs on blurry/corrupted photos; cryptic error traces; models that flag professional studio portraits as fake due to skin smoothing.

### 2.3 Forensic Machine Learning Engineer (Developer/Maintainer)
- **Objective:** Train, benchmark, debug, package, and maintain the model and inference pipeline under strict deadlines and resource budgets.
- **Key Needs:** Modular Python code (`preprocess.py`, `train.py`, `predict.py`, `evaluate.py`), clear configuration files (`preprocess.json`), locked seed scripts, automated unit tests, and lightweight dependency management.
- **Pain Points:** Complex C++ dependencies, CUDA driver mismatches, oversized models requiring enterprise GPUs, non-deterministic preprocessing pipelines.

### 2.4 End-User / Verification Subject (Applicant)
- **Objective:** Submit a portrait photograph to complete remote onboarding seamlessly.
- **Key Needs:** Fast response time (< 1.5 seconds), clear error messages if an image is unsupported or corrupted, and protection against unjustified automated rejection.
- **Pain Points:** Rejection without recourse; privacy violations from permanent retention of personal photos.

---

## 3. Dataset Architecture, Governance & Data Engineering

### 3.1 Dataset Selection Matrix & Analysis

To achieve a defensible classifier within a 3-day development timeframe, the system focuses on a curated, manageable subset of a premier benchmark dataset, with clear separation between internal validation and external testing.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               DATASET TAXONOMY & ROLE                                 │
├──────────────────────────────┬─────────────────────────────────────────────────────────┤
│ GenImage (Primary Training & │ • Structure: Paired real (ImageNet) & synthetic images  │
│ Internal Evaluation)         │ • Generator Families: Stable Diffusion v1.4/v1.5,       │
│                              │   Midjourney, ADM, GLIDE, Wukong, VQDM, BigGAN          │
│                              │ • MVP Subset: SD v1.4 / SD v1.5 vs ImageNet-val         │
│                              │ • Role: 70% Train, 15% Validation, 15% Locked Test       │
├──────────────────────────────┼─────────────────────────────────────────────────────────┤
│ Synthbuster (Held-Out        │ • Structure: 1,000 pristine real + synthetic generations│
│ External Evaluation)         │ • Generator Families: DALL-E 2, DALL-E 3, Midjourney v5,│
│                              │   Firefly, Stable Diffusion XL                          │
│                              │ • Role: Zero-shot cross-generator generalization test    │
├──────────────────────────────┼─────────────────────────────────────────────────────────┤
│ FaceForensics++ / Celeb-DF   │ • Role: Fallback conditional scope (Face-swap only)     │
│ (Alternative Biometric Scope)│ • Rule: Video-level isolation strictly enforced         │
└──────────────────────────────┴─────────────────────────────────────────────────────────┘
```

#### Benchmark Comparison & Selection Rationale

| Dataset | Modality | Primary Manipulations / Generators | Size | Access / Licensing | Selected Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GenImage** | Still Images | SD v1.4, SD v1.5, Midjourney, GLIDE, VQDM, ADM, BigGAN | ~1.3M images | Open research access, standard terms | **Primary Core Benchmark** (Use balanced subset of 5,000–10,000 pairs) |
| **Synthbuster** | Still Images | DALL-E 2/3, Midjourney v5, Firefly, SDXL | ~10,000 images | Zenodo open repository (CC-BY 4.0) | **External Generalization Benchmark** (Held-out test only) |
| **FaceForensics++** | Video Frames | Deepfakes, Face2Face, FaceSwap, NeuralTextures | ~1.8M frames | Research access form required | **Contingency Only** (If task shifted specifically to face-swapping) |
| **Celeb-DF v2** | Video Frames | Advanced deepfake face swaps | 5,639 videos | Academic request required | **Contingency Only** |
| **IMD2020** | Still Images | Splicing, inpainting, copy-move | 35,000 images | Academic open download | **Deferred** (Forensic localized manipulation, not full generation) |
| **CIFAKE** | Low-Res Images | Stable Diffusion v1.4 vs CIFAR-10 (32x32) | 120,000 images | Kaggle / Open | **Rejected for Identity** (32x32 resolution lacks facial forensic realism) |

### 3.2 Dataset Acquisition & Governance Protocol
1. **Acquisition Auditing:** Record dataset name, version identifier, direct download URL, download timestamp, cryptographic hash of raw archives, and original license terms.
2. **Immutable Raw Storage:** Store original files under `data/raw/` with read-only filesystem attributes. No pre-processing, cropping, or script modifications are permitted directly in `data/raw/`.
3. **Partitioned Processed Storage:** All standardized, downsampled, or curated subsets must be generated into `data/processed/` alongside an explicit generation manifest.
4. **Licensing Compliance:** Model weights and codebase must respect upstream licenses. Restricted raw dataset files must never be checked into public Git repositories (enforced via `.gitignore`).

### 3.3 Data Manifest Schema Specification
Every processed image is cataloged in a structured JSON/CSV manifest (`manifests/dataset_manifest.json`) adhering to the following schema:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "DatasetManifestEntry",
  "type": "object",
  "properties": {
    "asset_id": { "type": "string", "description": "Unique UUID4 or SHA-256 prefix" },
    "relative_path": { "type": "string", "description": "Relative path from project root" },
    "source_dataset": { "type": "string", "enum": ["GenImage", "Synthbuster", "Custom_Demo"] },
    "label": { "type": "integer", "enum": [0, 1], "description": "0: Real / Authentic, 1: Synthetic / Fake" },
    "generator_family": { "type": "string", "description": "e.g., StableDiffusion_v1.5, Midjourney, None_Real" },
    "split": { "type": "string", "enum": ["train", "val", "test", "external_test"] },
    "sha256": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
    "dimensions": {
      "type": "array",
      "items": { "type": "integer" },
      "minItems": 2,
      "maxItems": 2
    },
    "original_format": { "type": "string", "enum": ["JPEG", "PNG", "WEBP"] },
    "filesize_bytes": { "type": "integer" }
  },
  "required": ["asset_id", "relative_path", "source_dataset", "label", "split", "sha256", "dimensions"]
}
```

### 3.4 Data Cleansing, Sanitization & Leakage Prevention Rules
- **Near-Duplicate Decontamination:** Apply perceptual hashing (`imagehash.phash`) across all samples. Any pair with Hamming distance $\le 3$ must be audited; duplicates must reside in the exact same split to prevent train-to-test leakage.
- **Generator-Aware / Group-Aware Stratification:** If multiple generators exist in GenImage (e.g., SD v1.4, BigGAN), splits must be generator-stratified or generator-disjoint. Under generator-stratified splitting, the ratio of real-to-fake and generator family distributions is identical across train (70%), validation (15%), and test (15%).
- **Subject-Aware Grouping (FaceForensics / Video Fallback):** If facial manipulation video datasets are activated, frames derived from the same source subject or video clip must **never** cross split boundaries. Violation of this rule invalidates evaluation.
- **Class Balance Enforcement:** The training set must maintain an exact 50:50 class balance (or maximum 45:55 tolerance). If real images exceed synthetic samples, apply uniform random downsampling to the majority class and log the seed.

### 3.5 Dataset Split Allocation Protocol

```
Total Curated Dataset: N Samples (e.g., N = 10,000; 5,000 Real / 5,000 Synthetic)
 ├── Train Split: 70% (7,000 samples) ──────► Gradient updates, backpropagation, optimizer steps
 ├── Validation Split: 15% (1,500 samples) ──► Checkpoint selection, early stopping, hyperparameter tuning, temperature calibration
 └── Test Split: 15% (1,500 samples) ────────► LOCKED UNTIL SUBMISSION. Evaluated exactly once. Zero tuning.
 
External Generalization Set: Synthbuster (1,000 samples)
 └── Held-Out Generalization Split ──────────► Zero-shot test on unseen generators (DALL-E 3, Midjourney v5).
```

---

## 4. Machine Learning System Architecture & Modeling Strategy

### 4.1 Theoretical Foundation: Generative Artifact Footprints
Synthetic images generated by deep neural networks exhibit fundamental digital traces distinguishing them from natural optical captures:
1. **Spatial Artifacts:** Structural incoherence in complex micro-geometry (e.g., mismatched corneal reflections, distorted earlobes, asymmetric pupil boundaries, irregular hair strands, blending seams around skin edges).
2. **Frequency Domain Anomalies:** Upsampling layers (transposed convolutions, bilinear upscaling) in generative decoders introduce periodic grid artifacts and phase discontinuities visible in the 2D Fast Fourier Transform (FFT) and Discrete Cosine Transform (DCT) spectra.
3. **Color Space & Sensor Noise Inconsistencies:** Natural photographs contain physical camera sensor noise (Photo Response Non-Uniformity / PRNU) and chromatic aberrations across the Bayer CFA matrix. Diffusion models synthesize smooth, synthetic noise distributions that lack physical camera pipeline sensor characteristics.

### 4.2 Model Backbone Analysis & Selection

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          BACKBONE CANDIDATE EVALUATION                                 │
├─────────────────┬──────────┬──────────────┬───────────────┬────────────────────────────┤
│ Architecture    │ Params   │ FLOPs (224²) │ PyTorch Ready │ Forensic Fit / Suitability │
├─────────────────┼──────────┼──────────────┼───────────────┼────────────────────────────┤
│ EfficientNet-B0 │ 5.3 M    │ 0.39 GFLOPs  │ torchvision   │ HIGHEST: Compound scaling, │
│ (RECOMMENDED)   │          │              │ native        │ exceptional parameter/     │
│                 │          │              │               │ accuracy tradeoff, fast    │
├─────────────────┼──────────┼──────────────┼───────────────┼────────────────────────────┤
│ ResNet-18       │ 11.7 M   │ 1.82 GFLOPs  │ torchvision   │ HIGH: Simple residual      │
│ (FALLBACK)      │          │              │ native        │ topology, highly reliable, │
│                 │          │              │               │ standard baseline in paper │
├─────────────────┼──────────┼──────────────┼───────────────┼────────────────────────────┤
│ MobileNetV3-L   │ 5.4 M    │ 0.22 GFLOPs  │ torchvision   │ MEDIUM: Excellent speed,   │
│                 │          │              │ native        │ slightly lower spatial rep │
├─────────────────┼──────────┼──────────────┼───────────────┼────────────────────────────┤
│ ConvNeXt-Tiny   │ 28.6 M   │ 4.46 GFLOPs  │ torchvision   │ HIGH: Modern CNN, captures │
│                 │          │              │ native        │ subtle texture, heavier    │
├─────────────────┼──────────┼──────────────┼───────────────┼────────────────────────────┤
│ ViT-B/16        │ 86.6 M   │ 17.6 GFLOPs  │ torchvision   │ LOW FOR 3-DAY: High VRAM,  │
│                 │          │              │ native        │ data-hungry, slow training │
└─────────────────┴──────────┴──────────────┴───────────────┴────────────────────────────┘
```

**Architectural Decision:** Adopt **EfficientNet-B0** as the primary backbone, with **ResNet-18** as the guaranteed operational fallback. Both architectures are natively available in `torchvision.models` with pre-trained ImageNet-1k weights (`weights=EfficientNet_B0_Weights.DEFAULT`), eliminating external download risks and compile overhead.

### 4.3 Classification Head & Transfer Learning Protocol
The pretrained classification head (originally 1,000 classes) is replaced with a customized forensic binary classifier:

```
[EfficientNet-B0 Backbone (Features)] 
                 │ (Output: 1280-dim feature vector)
                 ▼
     [AdaptiveAvgPool2d(1x1)]
                 │
                 ▼
          [Flatten(start_dim=1)]
                 │
                 ▼
          [Dropout(p=0.3)]
                 │
                 ▼
        [Linear(1280 -> 256)]
                 │
                 ▼
              [SiLU()]
                 │
          [Dropout(p=0.2)]
                 │
                 ▼
         [Linear(256 -> 1)] ──► Raw Scalar Logit z
```

#### Staged Fine-Tuning Execution Protocol
1. **Stage 1 (Linear Probing / Feature Extractor Warmup):**
   - Freeze all convolutional backbone parameters ($\theta_{\text{backbone}}$).
   - Set requires_grad = False for stages 0 through 7.
   - Train classification head ($\theta_{\text{head}}$) only.
   - Learning Rate: $\eta = 1 \times 10^{-3}$, Optimizer: AdamW, Weight Decay: $1 \times 10^{-4}$.
   - Duration: 2 Epochs.
2. **Stage 2 (Selective Unfreezing & Fine-Tuning):**
   - Unfreeze top convolutional stages (stages 6 and 7 of EfficientNet-B0).
   - Set differential learning rates: $\eta_{\text{backbone}} = 5 \times 10^{-5}$, $\eta_{\text{head}} = 2 \times 10^{-4}$.
   - Cosine Annealing learning rate schedule down to $\eta_{\min} = 1 \times 10^{-6}$.
   - Early stopping monitored on Validation Macro-F1 with patience = 3 epochs.
   - Duration: 5–8 Epochs.

### 4.4 Deterministic Preprocessing Pipeline
To guarantee zero discrepancy between training and production inference, preprocessing must be strictly identical:

```python
# Canonical Pipeline Specification
Input Image (RGB)
    │
    ├─ Step 1: EXIF Transpose (Resolve physical rotation via PIL.ImageOps.exif_transpose)
    │
    ├─ Step 2: Ensure 3-Channel RGB (Convert 'L', 'RGBA', 'CMYK' -> 'RGB')
    │
    ├─ Step 3: Resize to 224 x 224 (Interpolation: BICUBIC or BILINEAR, antialias=True)
    │
    ├─ Step 4: Convert to Float Tensor in range [0.0, 1.0]
    │
    └─ Step 5: Channel-wise Normalization (Standard ImageNet):
               Mean: [0.485, 0.456, 0.406]
               Std:  [0.229, 0.224, 0.225]
```

### 4.5 Data Augmentation Policy (Forensic-Preserving)
Standard computer vision augmentations can inadvertently destroy high-frequency generative artifacts or introduce synthetic artifacts into real photos. The following strict rules apply:

| Permitted Augmentations | Forbidden Augmentations | Rationale |
| :--- | :--- | :--- |
| Random Horizontal Flip ($p=0.5$) | Extreme Random Cropping ($<80\%$) | Severe crops alter global facial aspect ratio and geometry |
| Mild Random Rotation ($\pm 10^\circ$) | Strong Color Jitter (Hue $> 0.1$) | Extreme color shifts mask chromatic aberration cues |
| Subtle JPEG Compression ($Q \in [70, 95]$, $p=0.3$) | Gaussian Blur ($\sigma > 2.0$) | Heavy blur wipes out diffusion texture grain entirely |
| Mild Brightness/Contrast Adjustment ($\pm 10\%$) | CutMix / MixUp | Blending real and fake creates synthetic boundary artifacts |

### 4.6 Loss Function & Optimization Strategy
- **Loss Formulation:** Binary Cross-Entropy with Logits:
  $$\mathcal{L}_{\text{BCE}}(z, y) = -\left[ y \cdot \log \sigma(z) + (1 - y) \cdot \log (1 - \sigma(z)) \right]$$
  where $z$ is the raw model logit, $\sigma(z) = \frac{1}{1 + e^{-z}}$, and $y \in \{0, 1\}$.
- **Label Smoothing:** Apply modest label smoothing ($\epsilon = 0.05$) to prevent over-confident gradient saturation on noisy training samples:
  $$y_{\text{smooth}} = y(1 - \epsilon) + 0.5\epsilon$$

### 4.7 Post-Processing: Calibration & Abstention Engine
Raw Softmax probabilities are notoriously overconfident and cannot be interpreted as posterior probabilities.
1. **Temperature Scaling (Platt Scaling):**  
   Post-training, optimize a scalar temperature parameter $T > 0$ on the frozen validation set by minimizing negative log-likelihood:
   $$p_{\text{calibrated}} = \sigma\left(\frac{z}{T}\right) = \frac{1}{1 + \exp\left(-\frac{z}{T}\right)}$$
   where $T$ is obtained via L-BFGS optimization on validation logits.
2. **Three-Tier Decision Engine with Abstention:**
   Instead of forcing a reckless binary choice, the inference engine implements explicit forensic bands:
   - **Tier 1 (Flagged Synthetic):** $p_{\text{calibrated}} \ge \tau_{\text{fake}}$ (Default: $\tau_{\text{fake}} = 0.65$)  
     *Classification:* "AI-Generated / Synthetic Traces Detected"
   - **Tier 2 (Verified Real-Consistent):** $p_{\text{calibrated}} \le \tau_{\text{real}}$ (Default: $\tau_{\text{real}} = 0.35$)  
     *Classification:* "Consistent with Natural Optical Capture"
   - **Tier 3 (Abstention / Inconclusive):** $0.35 < p_{\text{calibrated}} < 0.65$ OR input fails quality checks  
     *Classification:* "Inconclusive / Insufficient Discriminative Evidence"

---

## 5. Comprehensive Evaluation & Verification Framework

### 5.1 Metrics Suite (Complete Equations & Definitions)
Evaluation scripts must calculate and output the complete suite of statistical performance metrics:

1. **Confusion Matrix Elements:**
   - True Positives ($TP$): Correctly identified Synthetic images.
   - True Negatives ($TN$): Correctly identified Real images.
   - False Positives ($FP$): Natural Real images wrongly classified as Synthetic (Type I Error).
   - False Negatives ($FN$): Synthetic images wrongly classified as Real (Type II Error).

2. **Classification Metrics:**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
   $$\text{Precision (Fake)} = \frac{TP}{TP + FP}, \quad \text{Recall / TPR (Fake)} = \frac{TP}{TP + FN}$$
   $$\text{Specificity / TNR} = \frac{TN}{TN + FP}, \quad \text{False Positive Rate (FPR)} = \frac{FP}{FP + TN}$$
   $$F_{1\text{-score}} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
   $$\text{Macro } F_1 = \frac{F_{1\text{-real}} + F_{1\text{-fake}}}{2}$$

3. **Threshold-Independent Metrics:**
   - **ROC-AUC:** Area under the Receiver Operating Characteristic curve (True Positive Rate vs. False Positive Rate across all thresholds $\tau \in [0, 1]$).
   - **PR-AUC:** Area under the Precision-Recall curve.
   - **Equal Error Rate (EER):** The specific decision threshold where False Acceptance Rate equals False Rejection Rate ($FPR = FNR$).
   - **Expected Calibration Error (ECE):**
     $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
     evaluated across $M=10$ confidence bins.

### 5.2 Locked Evaluation Protocol
- **Evaluation Isolation:** The test split (`data/processed/test_manifest.json`) remains completely unread during training, validation tuning, and threshold selection.
- **Automated Metric Emission:** Execution of `python src/evaluate.py --config configs/preprocess.json --weights models/best_model.pth --manifest manifests/test_manifest.json` generates:
  - `reports/metrics.json` (Structured machine-readable performance metrics)
  - `reports/confusion_matrix.png` (High-resolution annotated heatmap)
  - `reports/roc_curve.png` and `reports/pr_curve.png`
  - `reports/calibration_curve.png` (Reliability diagram)

### 5.3 Stress Testing & Robustness Degradation Analysis
The evaluation protocol includes stress testing the model against synthetic degradations simulating transmission across the web:
1. **JPEG Compression Degradation:** Evaluate test accuracy and AUC across quality factors: $Q \in \{95, 85, 75, 65, 50\}$.
2. **Resolution Downscaling:** Downsample inputs to $\{160\times 160, 128\times 128, 96\times 96\}$ before bicubic restoration to test sensitivity to low-resolution uploads.
3. **Additive Gaussian Noise:** Evaluate performance under noise levels $\sigma \in \{0.01, 0.03, 0.05\}$.

### 5.4 Systematic Error Analysis Protocol
The system must generate an automated error catalog (`reports/error_analysis.json`) documenting:
- Top 10 False Positives (Real images assigned highest synthetic score).
- Top 10 False Negatives (Synthetic images assigned lowest synthetic score).
- Analysis of confounding visual attributes: studio lighting, digital airbrushing, high-ISO sensor grain, heavy compression artifacts.

---

## 6. Functional Requirements & Traceability Matrix

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        REQUIREMENTS TRACEABILITY MAP                                   │
├────────────┬─────────────────────────────┬───────────────────┬─────────────────────────┤
│ Module ID  │ Component                   │ Requirement Range │ Acceptance Focus        │
├────────────┼─────────────────────────────┼───────────────────┼─────────────────────────┤
│ SEC-00     │ Input Security & Validation │ FR-01 to FR-05    │ Buffer safety, magic    │
│            │                             │                   │ bytes, resource defense │
├────────────┼─────────────────────────────┼───────────────────┼─────────────────────────┤
│ PRE-00     │ Forensic Preprocessing      │ FR-06 to FR-10    │ EXIF audit, format norm │
├────────────┼─────────────────────────────┼───────────────────┼─────────────────────────┤
│ INF-00     │ Inference & Uncertainty     │ FR-11 to FR-16    │ PyTorch forward, calib, │
│            │                             │                   │ 3-tier decision engine  │
├────────────┼─────────────────────────────┼───────────────────┼─────────────────────────┤
│ UI-00      │ Presentation & Reporting    │ FR-17 to FR-22    │ Streamlit UI, cards,    │
│            │                             │                   │ audit logs, caveats     │
├────────────┼─────────────────────────────┼───────────────────┼─────────────────────────┤
│ EVAL-00    │ Offline Evaluation & Bench  │ FR-23 to FR-27    │ Locked test, plots, ECE │
├────────────┼─────────────────────────────┼───────────────────┼─────────────────────────┤
│ OPS-00     │ Reproducibility & Packaging │ FR-28 to FR-30    │ Seeds, requirements,    │
│            │                             │                   │ model card docs         │
└────────────┴─────────────────────────────┴───────────────────┴─────────────────────────┘
```

### 6.1 Input Security, File Ingestion & Safety (SEC-00)

- **FR-01: Supported MIME & Magic Byte Validation**  
  The system shall inspect the initial 16 bytes (magic numbers) of any uploaded file stream to confirm genuine file format. Supported signatures:
  - JPEG: `FF D8 FF`
  - PNG: `89 50 4E 47 0D 0A 1A 0A`  
  Files whose extension (.jpg) contradicts their internal magic bytes, or files containing executable or polyglot headers, must be rejected immediately with an HTTP 400 / descriptive validation error.

- **FR-02: Strict File Size Boundaries**  
  The system shall enforce an absolute file size ceiling of 15.0 MB and a minimum payload floor of 1.0 KB. Oversized uploads must be blocked prior to reading the full stream into memory.

- **FR-03: Decompression Bomb & Dimensional Limits Defense**  
  To protect against decompression bomb denial-of-service attacks (e.g., zip bombs, 100,000 $\times$ 100,000 pixel sparse images), the system shall:
  - Configure `PIL.Image.MAX_IMAGE_PIXELS = 89_478_485` (or lower: maximum dimensions $4096 \times 4096$ pixels).
  - Inspect image dimensions via header decoding (`Image.open(stream)` without calling `.load()`).
  - Reject images with width or height $< 64$ pixels (insufficient facial/forensic information) or $> 4096$ pixels.

- **FR-04: Non-Trust of Client-Supplied Metadata**  
  The system shall completely ignore the client-provided HTTP `Content-Type` header and file extension when determining decode strategy. The filename shall be sanitized using alphanumeric whitelisting (`[a-zA-Z0-9_.-]`), discarding any directory path traversals (`../`).

- **FR-05: Exception Masking & Zero Internal Leakage**  
  The system shall catch all underlying decode exceptions (`PIL.UnidentifiedImageError`, `IOError`, `MemoryError`) and return sanitized, human-readable error messages. Internal filesystem paths, server stack traces, and environment variables must never be returned to the client UI.

### 6.2 Preprocessing & Forensic Inspection (PRE-00)

- **FR-06: EXIF Metadata Harvesting & Preservation**  
  The system shall inspect and extract standard EXIF tags (Make, Model, Software, DateTimeOriginal, ExposureTime, FNumber, ISO, GPSInfo) and ICC color profile tags before any pixel transformations. Extracted metadata shall be formatted for UI display and forensic logging.

- **FR-07: Deterministic EXIF Orientation Correction**  
  The system shall apply `PIL.ImageOps.exif_transpose()` to ensure portrait images with EXIF orientation flags (e.g., Orientation=6 for 90° clockwise rotation) are correctly oriented in pixel space prior to resizing.

- **FR-08: Color Space Standardization**  
  The system shall convert any multi-channel input (e.g., RGBA, Grayscale 'L', CMYK, palette-based 'P') into standard 3-channel RGB. Transparency layers in PNG images must be composited over a solid white background ($255, 255, 255$) rather than discarding alpha abruptly.

- **FR-09: Standardized Rescaling & Interpolation**  
  The system shall scale images to the exact target resolution ($224 \times 224$ pixels) using Bilinear or Bicubic interpolation with anti-aliasing enabled (`Image.Resampling.BILINEAR`, `antialias=True`).

- **FR-10: Tensor Conversion & Channel Normalization**  
  The normalized tensor must be formatted as float32 in shape `[1, 3, 224, 224]`, normalized using ImageNet standard constants:
  - $\mu = [0.485, 0.456, 0.406]$
  - $\sigma = [0.229, 0.224, 0.225]$

### 6.3 Inference, Calibration & Decision Engine (INF-00)

- **FR-11: Checkpoint Verification & Model Loading**  
  The inference engine shall load the model weights from a local path (`models/best_model.pth`). At startup, the system shall verify:
  - File existence and non-zero byte size.
  - Model architecture parameter match against `configs/preprocess.json`.
  - Checkpoint state dictionary keys match the model definition.
  - Explicit execution of `model.eval()` to deactivate Dropout and Freeze BatchNorm running statistics.

- **FR-12: Inference Execution & Hardware Selection**  
  Inference must execute dynamically on CUDA GPU if available (`torch.cuda.is_available()`), falling back gracefully to standard CPU. All inference operations must run inside a `torch.no_grad()` context block.

- **FR-13: Execution Latency Profiling**  
  The system shall measure and report inference latency in milliseconds ($t_{\text{infer}}$), capturing pure neural forward pass duration separately from image decoding and preprocessing overhead.

- **FR-14: Model Score Computation & Temperature Calibration**  
  The raw output logit $z \in (-\infty, \infty)$ shall be scaled using the calibrated temperature $T$ saved in `configs/preprocess.json`, followed by sigmoid activation to produce the normalized model score $s \in [0.0, 1.0]$:
  $$s = \frac{1}{1 + \exp(-z / T)}$$

- **FR-15: Explicit Abstention State (Inconclusive Decision Engine)**  
  The system shall evaluate the calibrated score $s$ against predefined thresholds:
  - If $s \ge 0.65$: Classify as `AI-Generated / Synthetic`.
  - If $s \le 0.35$: Classify as `Real / Authentic Consistent`.
  - If $0.35 < s < 0.65$: Classify as `Inconclusive / Abstain`.  
  The system must never force a binary prediction when confidence lies within the ambiguity margin.

- **FR-16: Low-Quality Abstention Override**  
  If the input image has native resolution $< 128 \times 128$ or exhibits extreme compression degradation (estimated via Laplacian blur variance $< 100.0$), the system shall trigger an automatic abstention state: `Unreliable Input - Quality Insufficient for Forensic Evaluation`.

### 6.4 User Interface & Presentation Layer (UI-00)

- **FR-17: Single & Batch Upload Interface**  
  The Streamlit UI (`app.py`) shall provide an intuitive drag-and-drop file upload zone accepting JPG, JPEG, and PNG files.

- **FR-18: Visual Preview & Dimensional Inspect Panel**  
  Upon upload, the UI shall render the image preview alongside vital forensic telemetry: original filename, file size (KB/MB), native dimensions (Width $\times$ Height), aspect ratio, color mode, and hash (SHA-256).

- **FR-19: Forensic Assessment Display Card**  
  The analysis view shall prominently present:
  - High-level verdict with color-coded visual indicator:
    * Red Badge: `Synthetic Traces Detected`
    * Green Badge: `Consistent with Natural Capture`
    * Amber Badge: `Inconclusive / Manual Review Required`
  - Model Confidence Score expressed as a percentage with explicit calibration caveat.
  - Latency badge (e.g., `Inference: 42ms | Total: 180ms`).

- **FR-20: Forensic Caveat & Disclaimers Box**  
  Every prediction screen must render an explicit, prominent disclaimer:
  > **Forensic Notice:** This score reflects machine learning statistical consistency against trained generator patterns. It does NOT constitute legal proof of authenticity or fraud. Absence of synthetic traces does not prove real-world identity or liveness.

- **FR-21: Curated Demo Gallery Showcase**  
  The application shall feature a 1-click "Load Demo Sample" gallery containing 8 pre-loaded, diverse images:
  - 2 $\times$ High-Quality Natural Portraits (ImageNet / Open-access photography)
  - 2 $\times$ Stable Diffusion Generations (Photorealistic portraits)
  - 2 $\times$ Midjourney v5/v6 Generations (Subtle diffusion artifacts)
  - 1 $\times$ Low-resolution / Highly compressed real image (Demonstrating abstention)
  - 1 $\times$ Edge-case non-face or non-standard aspect ratio sample

- **FR-22: Audit Report Export**  
  The UI shall provide a 1-click button to download a JSON forensic audit summary containing timestamp, SHA-256, EXIF summary, model version, raw logit, calibrated score, and decision verdict.

### 6.5 Offline Evaluation & Benchmark Automation (EVAL-00)

- **FR-23: Automated Evaluation CLI Script**  
  The system shall include `src/evaluate.py` capable of loading the locked test manifest and running full batch evaluation without UI dependencies.

- **FR-24: Comprehensive Metric Export (`metrics.json`)**  
  The evaluation script shall compute and save a complete metrics record including: total sample count, class balance (real count, fake count), TP, FP, TN, FN, Accuracy, Precision, Recall, Specificity, F1-Score, Macro-F1, ROC-AUC, PR-AUC, and ECE.

- **FR-25: High-Resolution Graphic Plot Generation**  
  The evaluation harness shall generate:
  - Annotated Confusion Matrix (`reports/confusion_matrix.png`)
  - ROC Curve with AUC score indicator (`reports/roc_curve.png`)
  - Precision-Recall Curve (`reports/pr_curve.png`)
  - Confidence Calibration Reliability Curve (`reports/calibration_curve.png`)

- **FR-26: Cross-Generator Generalization Evaluation**  
  The evaluation script shall support an optional `--external` flag to evaluate the locked model against the external Synthbuster test set, outputting a separate generalization report (`reports/generalization_metrics.json`).

- **FR-27: Automated Error Case Extraction**  
  The script shall automatically copy the top 5 worst False Positives and top 5 worst False Negatives into `reports/failures/` with diagnostic overlay text for post-mortem analysis.

### 6.6 Reproducibility, Configuration & Documentation (OPS-00)

- **FR-28: Global Deterministic Seeding**  
  All scripts (`train.py`, `evaluate.py`, `preprocess.py`) shall enforce deterministic random seeding across:
  - Python `random.seed(seed)`
  - NumPy `np.random.seed(seed)`
  - PyTorch `torch.manual_seed(seed)`
  - CUDA `torch.cuda.manual_seed_all(seed)`
  - `torch.backends.cudnn.deterministic = True` (Default seed: `42`).

- **FR-29: Centralized Preprocessing & Model Configuration (`preprocess.json`)**  
  All constants (input resolution, normalization mean/std, model architecture name, temperature parameter, decision thresholds, class index mapping) shall be maintained in a single versioned file: `configs/preprocess.json`.

- **FR-30: Comprehensive Model Card Documentation (`model_card.md`)**  
  The repository shall provide a complete model card detailing: model architecture, parameter count, training dataset details, license, intended use, out-of-scope applications, demographic considerations, and benchmark performance.

---

## 7. Non-Functional Requirements (NFR)

### 7.1 Performance & Latency Budgets

```
┌────────────────────────────────────────────────────────────────────────┐
│                        LATENCY BUDGET BREAKDOWN                        │
├──────────────────────────┬──────────────┬──────────────┬───────────────┤
│ Stage                    │ Target (P50) │ Target (P95) │ Hard Ceiling  │
├──────────────────────────┼──────────────┼──────────────┼───────────────┤
│ File Ingest & Magic Byte │ 5 ms         │ 15 ms        │ 50 ms         │
│ EXIF Inspection          │ 10 ms        │ 25 ms        │ 100 ms        │
│ Resize & Tensor Convert  │ 20 ms        │ 45 ms        │ 150 ms        │
│ GPU Model Inference      │ 25 ms        │ 45 ms        │ 100 ms        │
│ CPU Model Inference      │ 120 ms       │ 280 ms       │ 800 ms        │
│ Post-Process & Score UI  │ 10 ms        │ 25 ms        │ 50 ms         │
├──────────────────────────┼──────────────┼──────────────┼───────────────┤
│ Total End-to-End (GPU)   │ ~70 ms       │ ~155 ms      │ < 450 ms      │
│ Total End-to-End (CPU)   │ ~165 ms      │ ~390 ms      │ < 1150 ms     │
└──────────────────────────┴──────────────┴──────────────┴───────────────┘
```

- **Throughput:** In offline batch evaluation, the system must process $\ge 50\text{ images/second}$ on an NVIDIA RTX 3060/4060 GPU with batch size = 32.
- **Memory Footprint:** 
  - Host RAM consumption during active inference must remain below 1.2 GB.
  - Dedicated VRAM consumption during inference must remain below 1.5 GB.
  - Training VRAM consumption with batch size 32 must fit within 4.0 GB VRAM.

### 7.2 Security, System Hardening & Isolation
- **Buffer Overflow & Memory Exploitation:** Image decoding must rely solely on audited, modern Pillow implementations (`>=10.2.0`). Custom binary C++ decoders are prohibited.
- **Resource Exhaustion Defense:** Total execution timeout for any single uploaded image is capped at 5.0 seconds. If processing hangs, the thread is terminated and an error returned.
- **Least Privilege Execution:** The application must run under an unprivileged user space without requiring root or administrator privileges.
- **Network Isolation:** Core inference and validation must execute 100% locally without dispatching outbound HTTP requests or phoning home to external cloud APIs.

### 7.3 Data Privacy & Governance
- **Zero Permanent Retention:** By default, user-uploaded images must be processed ephemerally in RAM or temporary scratch storage, deleted immediately after the session turn ends or after maximum 10 minutes.
- **No PII Persistence:** System logs must store only cryptographic hashes (SHA-256), image dimensions, processing times, and classification scores. Raw image bytes and personal identity metadata must never be written to persistent log files.
- **Dataset Privacy:** Demo galleries and evaluation manifests must only contain publicly released, open-access, or fully synthetic images. No private personal data shall be bundled.

### 7.4 Reliability & Fault Tolerance
- **Graceful Degradation:** In the event of GPU unavailability or CUDA driver incompatibility, the system must automatically fall back to CPU execution without throwing unhandled exceptions.
- **Corrupted Input Containment:** A batch containing 1 corrupted image must not crash the entire evaluation script; the script must log the error, record the asset ID as `CORRUPTED`, and proceed with remaining items.

### 7.5 Portability & System Dependencies
- **Target OS Compatibility:** Fully compatible with Windows 10/11 (PowerShell), Linux (Ubuntu 20.04+), and macOS (Apple Silicon via MPS or CPU).
- **Python Version Pinning:** Validated against Python 3.10 and 3.11.
- **Dependency Isolation:** All required Python packages must be locked with strict version constraints in `requirements.txt`.

---

## 8. System Architecture & Component Design

### 8.1 High-Level Architecture Diagram

```mermaid
flowchart TD
    subgraph UI_Layer ["Presentation & Interaction Layer (Streamlit / CLI)"]
        UI["Web UI Dashboard (app.py)"]
        CLI["Evaluation & Training CLI (src/evaluate.py, src/train.py)"]
        Demo["Curated Demo Gallery (assets/demo/)"]
    end

    subgraph Security_Gate ["Security & Validation Layer (src/security.py)"]
        Magic["Magic Byte Inspector (JPEG/PNG)"]
        Limits["Dimension & Size Enforcer (64px to 4096px, <15MB)"]
        Sanitize["Header & Filename Sanitizer"]
    end

    subgraph Preprocess_Layer ["Forensic Preprocessing Layer (src/preprocess.py)"]
        EXIF["EXIF & Metadata Harvester"]
        Orient["Orientation Normalizer (EXIF Transpose)"]
        Resize["Deterministic Resize (224x224 Bicubic)"]
        Norm["Tensor Normalizer (ImageNet Constants)"]
    end

    subgraph Inference_Engine ["Neural Engine & Post-Processing (src/predict.py)"]
        Model["PyTorch Backbone (EfficientNet-B0 / ResNet-18)"]
        Weights[("Checkpoint: models/best_model.pth")]
        Config[("Config: configs/preprocess.json")]
        Calib["Temperature Calibration Layer (T)"]
        Abstain["Abstention & Decision Engine (0.35 / 0.65)"]
    end

    subgraph Audit_Reporting ["Audit & Evaluation Reporting (reports/)"]
        Metrics[("reports/metrics.json")]
        Plots[("Confusion Matrix & ROC Curves")]
        ModelCard[("reports/model_card.md")]
    end

    UI --> Security_Gate
    CLI --> Security_Gate
    Demo --> Security_Gate

    Security_Gate -->|Valid Input| Preprocess_Layer
    Security_Gate -->|Invalid Input| UI

    Preprocess_Layer --> Inference_Engine
    Weights --> Model
    Config --> Inference_Engine

    Inference_Engine --> UI
    Inference_Engine --> Audit_Reporting
```

### 8.2 Image Processing Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> UploadReceived: File Stream Ingested
    
    UploadReceived --> ValidatingSecurity: Check File Size & Magic Bytes
    ValidatingSecurity --> RejectedMalformed: Magic bytes != JPEG/PNG OR Size > 15MB
    ValidatingSecurity --> CheckingDimensions: Magic bytes Valid
    
    CheckingDimensions --> RejectedDimensions: Dim < 64px OR Dim > 4096px
    CheckingDimensions --> ExtractingMetadata: Dimensions within Bounds
    
    ExtractingMetadata --> Preprocessing: Read EXIF / Sanitize
    Preprocessing --> QualityCheck: EXIF Orient + Resize 224x224 + RGB Ensure
    
    QualityCheck --> AbstainLowQuality: Blur Variance < 100 OR Low Information
    QualityCheck --> RunningInference: Quality Acceptable
    
    RunningInference --> CalibratingScore: Forward Pass -> Logit z
    CalibratingScore --> EvaluatingThresholds: Scaled Logit z/T -> Sigmoid
    
    EvaluatingThresholds --> FlaggedSynthetic: Calibrated Score >= 0.65
    EvaluatingThresholds --> VerifiedReal: Calibrated Score <= 0.35
    EvaluatingThresholds --> AbstainInconclusive: 0.35 < Score < 0.65
    
    FlaggedSynthetic --> GeneratingReport: Format Telemetry & Disclaimers
    VerifiedReal --> GeneratingReport: Format Telemetry & Disclaimers
    AbstainInconclusive --> GeneratingReport: Format Telemetry & Disclaimers
    AbstainLowQuality --> GeneratingReport: Return Quality Warning
    RejectedMalformed --> GeneratingReport: Return Security Error
    RejectedDimensions --> GeneratingReport: Return Dimension Error

    GeneratingReport --> [*]: Render in UI / Return Response
```

### 8.3 Recommended Repository Layout & Code Organization

```
Deepfake Image Detection/
│
├── docs/
│   └── prd.md                           # This exhaustive Product Requirements Document
│
├── configs/
│   ├── preprocess.json                  # Canonical normalization, dimensions, threshold configs
│   └── train_config.json                # Epochs, learning rates, batch sizes, seed configs
│
├── assets/
│   └── demo/                            # Curated 6-10 sample demo images
│       ├── real_01.jpg                  # Authentic portrait sample 1
│       ├── real_02.jpg                  # Authentic portrait sample 2
│       ├── fake_sd_01.jpg               # Stable Diffusion portrait sample 1
│       ├── fake_sd_02.jpg               # Stable Diffusion portrait sample 2
│       ├── fake_mj_01.jpg               # Midjourney v5 portrait sample
│       ├── edge_lowres.jpg              # Low-resolution abstention test sample
│       └── edge_corrupt.png             # Malformed header test sample
│
├── data/
│   ├── raw/                             # Read-only original downloaded datasets (NOT committed)
│   ├── processed/                       # Curated train/val/test image directories
│   └── manifests/
│       ├── dataset_manifest.json        # Master manifest cataloging all processed images
│       ├── train_manifest.json          # 70% Training split manifest
│       ├── val_manifest.json            # 15% Validation split manifest
│       ├── test_manifest.json           # 15% Locked test split manifest
│       └── external_manifest.json       # Held-out Synthbuster manifest
│
├── models/
│   └── best_model.pth                   # Trained PyTorch model checkpoint
│
├── reports/
│   ├── metrics.json                     # Complete locked test metrics
│   ├── confusion_matrix.png             # Annotated confusion matrix graphic
│   ├── roc_curve.png                    # ROC-AUC evaluation curve
│   ├── calibration_curve.png            # Reliability diagram
│   ├── error_analysis.json              # Catalog of top false positives/negatives
│   └── model_card.md                    # Structured forensic model card
│
├── src/
│   ├── __init__.py
│   ├── security.py                      # Magic byte, decompression bomb & size validation
│   ├── preprocess.py                    # EXIF extraction, orientation, resize, normalization
│   ├── dataset.py                       # PyTorch Dataset implementation reading from manifests
│   ├── model.py                         # EfficientNet-B0 / ResNet-18 model architecture definition
│   ├── train.py                         # Staged transfer learning training script
│   ├── calibration.py                   # Temperature scaling & threshold calibration
│   ├── predict.py                       # Core inference engine & abstention decider
│   └── evaluate.py                      # Offline evaluation, metric calculation & plot export
│
├── app.py                               # Streamlit interactive forensic dashboard
├── requirements.txt                     # Pinned Python package dependencies
├── .gitignore                           # Excludes data/, models/*.pth, __pycache__/
└── README.md                            # Complete setup, replication, and operation guide
```

---

## 9. Comprehensive Edge Cases, Failure Modes & Mitigations

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        EDGE CASE TAXONOMY & MITIGATION                                 │
├────────────────────────────────────────┬───────────────────────────────────────────────┤
│ Input Anomalies & Corner Cases         │ Architectural Defense / System Response       │
├────────────────────────────────────────┼───────────────────────────────────────────────┤
│ 1. Decompression Bomb Attack           │ PIL MAX_IMAGE_PIXELS cap + stream header check│
│ 2. Polyglot / Masked Script Files      │ 16-byte magic number verification             │
│ 3. Micro-Resolution Images (<64x64)    │ Pre-inference validation rejection (FR-03)    │
│ 4. Ultra-Low Resolution (64px to 127px)│ Abstention: Insufficient Quality (FR-16)      │
│ 5. Extreme Aspect Ratios (e.g., 20:1)  │ Pad with white borders before resize          │
│ 6. Single-Channel / Grayscale (Mode L) │ Replicate channel across 3 RGB planes         │
│ 7. Embedded Alpha Channel (PNG RGBA)   │ Alpha composite over solid white canvas       │
│ 8. Non-Standard Color Profiles (CMYK)  │ Convert CMYK -> RGB via Pillow engine         │
│ 9. Malicious EXIF Payloads (XSS)       │ Strict sanitization; treat EXIF as raw string │
│ 10. Heavy Social Media Re-compression  │ Calibrated Abstention (Uncertainty Band)      │
│ 11. Studio Lighting / Airbrushed Skin  │ Forensic Disclaimer (Score Caveat FR-20)      │
│ 12. Moiré Patterns from Screen Photos  │ Documented Limitation / Model Card disclosure │
└────────────────────────────────────────┴───────────────────────────────────────────────┘
```

### Detailed Edge Case Breakdown & Resolution Protocols

1. **Extreme Decompression Bombs:**
   - *Threat:* An attacker uploads a $500\text{ KB}$ compressed PNG file that expands in memory into a $40,000 \times 40,000$ uncompressed 24-bit bitmap ($4.8\text{ GB}$), crashing the host process.
   - *Mitigation:* Before calling `Image.open().load()`, read image dimensions via `Image.open(stream)`. Enforce `width * height <= 4096 * 4096`. Terminate decode immediately if threshold is exceeded.

2. **Polyglot / Executable Renaming:**
   - *Threat:* Malicious binary or HTML script renamed to `avatar.jpg`.
   - *Mitigation:* `src/security.py` parses the binary header directly. If the first 3 bytes do not match `FF D8 FF` (JPEG) or 8 bytes do not match `89 50 4E 47 0D 0A 1A 0A` (PNG), the upload is immediately rejected with `Validation Error: File signature does not match permitted image formats`.

3. **Color Space & Channel Variations:**
   - *Case A (Grayscale 'L'):* Portrait scans uploaded in monochrome. Resolution: Convert to RGB by repeating the single luminance channel across R, G, and B.
   - *Case B (Transparent RGBA):* Cutout portrait photos with transparent backgrounds. Resolution: Create a pure white RGB canvas and composite the RGBA image onto it using the alpha channel as a mask.
   - *Case C (Print Profile CMYK):* ID scans in CMYK mode. Resolution: Convert via `image.convert('RGB')` using standard ICC sRGB color space conversion.

4. **Aspect Ratio Distortion:**
   - *Threat:* A $2000 \times 100$ panoramic strip forcefully squished into a $224 \times 224$ square creates unnatural horizontal warping artifacts, fooling the classifier into predicting synthetic distortion.
   - *Mitigation:* Calculate aspect ratio $AR = \max(W/H, H/W)$. If $AR > 3.0$, the system applies letterboxing/pillarboxing (padding with neutral gray $128, 128, 128$) to square bounds prior to resizing, or rejects the image if $AR > 5.0$.

5. **Heavy Social Media Re-compression (WhatsApp / Facebook Proxies):**
   - *Phenomenon:* Platforms apply aggressive lossy JPEG re-compression (down to Quality 60) and chroma subsampling (4:2:0), obliterating high-frequency diffusion artifacts.
   - *Mitigation:* The calibration and abstention engine detects attenuated high-frequency variance; the score falls naturally into the middle band ($0.35 \le s \le 0.65$), triggering an explicit `Inconclusive - Compression Obscured Forensic Signals` recommendation.

6. **Airbrushed Natural Skin vs. GAN Smoothness:**
   - *Phenomenon:* High-end DSLR portraits taken with glamor studio lighting and beauty retouching have smooth skin surfaces resembling StyleGAN or Midjourney synthesis.
   - *Mitigation:* The model is evaluated on retouched real subsets during error analysis; the UI prominently explains that cosmetic skin smoothing may yield elevated synthetic scores.

7. **Recaptured Display Screens (Moiré Artifacts):**
   - *Phenomenon:* An applicant holds a smartphone displaying a photograph and photographs the screen with another device. The physical grid of LCD/OLED pixels introduces high-frequency interference patterns (Moiré).
   - *Mitigation:* Explicitly cataloged in `model_card.md` as an out-of-scope presentation attack condition. The system advises the operator that screen-recapture detection requires dedicated Presentation Attack Detection (PAD) algorithms.

---

## 10. Responsible AI, Ethics, Legal & Forensic Integrity

### 10.1 Prohibited Consequential Decisions (Anti-Harm Mandate)
- **No Autonomous Denial of Service:** This software must **never** be configured as an autonomous "hard gate" to unilaterally deny individuals access to financial accounts, credit, government services, or identity verification.
- **Human-in-the-Loop Mandate:** Every automated determination indicating synthetic media must serve solely as a diagnostic triage alert for a qualified human reviewer.
- **Presumption of Authenticity:** In forensic jurisprudence, missing EXIF metadata or stripped camera tags do not establish criminal manipulation, as virtually all consumer social platforms (Twitter, WhatsApp, Instagram, Telegram) routinely strip EXIF metadata by default during transit.

### 10.2 Demographic Fairness & Disparate Impact Considerations
- **Fairness Risk:** Facial analysis models trained predominantly on Western or celebrity datasets can exhibit unequal error rates across demographic cohorts (skin tone, gender, age).
- **Evaluation Obligation:** While comprehensive demographic debiasing is beyond a 3-day MVP scope, the project team must:
  - Document the demographic composition of the underlying dataset (GenImage / ImageNet).
  - Explicitly disclose demographic limitations in `reports/model_card.md`.
  - Prohibit marketing or deploying the tool as an unbiased universal identity auditor.

### 10.3 Regulatory & Standards Alignment
- **NIST AI Risk Management Framework (AI RMF 1.0):** Adheres to the core tenets of *Govern, Map, Measure, Manage* by documenting limitations, quantifying calibration error, and implementing abstention mechanisms.
- **European Union Artificial Intelligence Act (EU AI Act):** High-risk AI systems in identity verification and biometrics require transparency, logging, human oversight, and data governance. The system complies by providing model cards, audit trails, and zero autonomous denial.
- **C2PA / Content Authenticity Initiative (CAI):** Recognizes that probabilistic deepfake detection is inherently reactive; true media provenance requires end-to-end cryptographic asset signing (C2PA). The model card highlights C2PA as the long-term industry complementary standard.

---

## 11. Three-Day Execution Roadmap & Implementation Plan

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              3-DAY DELIVERY TIMELINE                                   │
├────────────────────┬──────────────────────────────────┬────────────────────────────────┤
│ Timeline           │ Core Objectives                  │ Verified Exit Criteria         │
├────────────────────┼──────────────────────────────────┼────────────────────────────────┤
│ DAY 1: Foundation  │ • Setup directories & configs    │ • Dataset verified & manifest  │
│ (Data & Baseline)  │ • Acquire GenImage subset        │   generated (train/val/test)   │
│                    │ • Implement manifest generator   │ • Train pipeline runs 1 epoch  │
│                    │ • Implement model.py & train.py  │ • Validation loop executes     │
│                    │ • Complete Stage 1 & 2 training  │ • models/best_model.pth saved  │
├────────────────────┼──────────────────────────────────┼────────────────────────────────┤
│ DAY 2: Application │ • Implement src/security.py      │ • End-to-end inference works   │
│ (Core Pipeline)    │ • Implement src/preprocess.py    │ • Abstention engine functional │
│                    │ • Implement src/predict.py       │ • Streamlit UI launches        │
│                    │ • Calibrate temperature T        │ • Sample image uploads render  │
│                    │ • Build app.py Streamlit UI      │   clean score & metadata       │
│                    │ • Assemble 8 curated demo cases  │ • Demo gallery operational     │
├────────────────────┼──────────────────────────────────┼────────────────────────────────┤
│ DAY 3: Validation  │ • Execute src/evaluate.py on     │ • reports/metrics.json created │
│ (Evaluation & Doc) │   locked test set                │ • Confusion matrix & ROC plots │
│                    │ • Generate confusion matrix &    │ • Automated error catalog      │
│                    │   reliability diagrams           │ • Clean repo reproducibility   │
│                    │ • Run generalization on external │ • Complete README & Model Card │
│                    │ • Author reports/model_card.md   │ • Defense presentation ready   │
│                    │ • Finalize README.md & rehearse  │                                │
└────────────────────┴──────────────────────────────────┴────────────────────────────────┘
```

### Detailed Daily Milestone Breakdown

#### Day 1 — Data Pipeline, Architecture & Model Training
- **Hour 01–02:** Initialize workspace structure (`configs/`, `src/`, `data/`, `reports/`, `models/`, `assets/demo/`). Create virtual environment and install pinned `requirements.txt`.
- **Hour 03–05:** Acquire GenImage balanced subset (5,000 real images from ImageNet-val, 5,000 synthetic images from Stable Diffusion v1.4/v1.5). Verify SHA-256 hashes.
- **Hour 06–07:** Build manifest generator. Split data: 70% Train (7,000), 15% Validation (1,500), 15% Locked Test (1,500). Verify generator stratification and zero overlap.
- **Hour 08–10:** Implement `src/model.py` (EfficientNet-B0 with custom classifier head) and `src/dataset.py` (PyTorch Dataset reading from manifest).
- **Hour 11–14:** Execute `src/train.py`. Run Phase 1 (warmup head, 2 epochs). Run Phase 2 (fine-tuning top layers, 6 epochs) with early stopping on validation Macro-F1.
- **Exit Gate 1:** `models/best_model.pth` exists, runs validation inference, achieves validation accuracy $> 85\%$.

#### Day 2 — Ingestion Security, Inference Engine & Interactive UI
- **Hour 15–17:** Implement `src/security.py`: magic byte inspection, PIL decompression bomb boundaries, dimension checks ($64 \le \dim \le 4096$), and sanitized error handling.
- **Hour 18–20:** Implement `src/preprocess.py`: EXIF metadata extraction, EXIF orientation normalization, RGB conversion, bilinear resizing to $224 \times 224$, and ImageNet normalization.
- **Hour 21–23:** Implement `src/calibration.py` to calculate temperature parameter $T$ on validation logits. Implement `src/predict.py` with 3-tier decision engine (Synthetic $\ge 0.65$, Real $\le 0.35$, Inconclusive).
- **Hour 24–27:** Build `app.py` using Streamlit. Implement drag-and-drop upload, image preview, metadata display, prediction badge, calibrated score gauge, and forensic disclaimers.
- **Hour 28–30:** Curate and populate `assets/demo/` with 8 diverse test cases (authentic, SD, Midjourney, compressed, corrupted). Verify 1-click loading.
- **Exit Gate 2:** User can launch `streamlit run app.py`, upload arbitrary images or select demo samples, and receive accurate, calibrated forensic telemetry end-to-end.

#### Day 3 — Rigorous Evaluation, Error Analysis & Submission Package
- **Hour 31–33:** Implement `src/evaluate.py`. Unlock test manifest (`manifests/test_manifest.json`) and run batch evaluation. Calculate all required metrics.
- **Hour 34–36:** Generate high-resolution plots: `reports/confusion_matrix.png`, `reports/roc_curve.png`, `reports/pr_curve.png`, and `reports/calibration_curve.png`. Write `reports/metrics.json`.
- **Hour 37–38:** Execute automated error analysis to extract top False Positives and False Negatives into `reports/error_analysis.json`.
- **Hour 39–41:** Run cross-generator generalization evaluation on Synthbuster subset (held-out DALL-E 3 / Midjourney v5). Record zero-shot performance drop.
- **Hour 42–45:** Author comprehensive documentation: write `reports/model_card.md` and complete `README.md` with step-by-step setup and replication commands.
- **Hour 46–48:** Final sanity audit in a fresh clean environment. Verify complete reproducibility from scratch. Rehearse academic demonstration.
- **Exit Gate 3:** Clean, defensible submission package with passing tests, reproducible evaluation, verified model card, and zero unhandled errors.

---

## 12. Verification & Testing Protocol

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        COMPREHENSIVE TEST SUITE MATRIX                                 │
├────────────────────┬──────────────────────────────────┬────────────────────────────────┤
│ Test Category      │ Target Components                │ Verification Method & Tool     │
├────────────────────┼──────────────────────────────────┼────────────────────────────────┤
│ Security & Ingest  │ Magic bytes, file size, bomb cap │ PyTest: Malformed & giant mock │
│ Preprocessing      │ EXIF transpose, RGB, resize norm │ PyTest: Rotated/RGBA test imgs │
│ Inference Engine   │ Tensor shape, eval mode, latency │ PyTest: Forward pass validation│
│ Calibration        │ Temperature scaling, bounds      │ Mathematical bounds assertions │
│ UI & Interaction   │ Streamlit dashboard, demo click  │ Interactive manual & automated │
│ Benchmark Pipeline │ Held-out test evaluation script  │ CLI: src/evaluate.py execution │
└────────────────────┴──────────────────────────────────┴────────────────────────────────┘
```

### 12.1 Automated Unit & Integration Tests (`tests/`)

1. **Security & Input Validation Tests (`tests/test_security.py`):**
   - Test `validate_image_buffer()` accepts standard JPEG (`FF D8 FF`) and PNG (`89 50 4E 47`).
   - Test `validate_image_buffer()` raises `ValidationError` when given a text/executable file renamed to `.jpg`.
   - Test dimensional boundaries: Reject $32 \times 32$ image; reject $5000 \times 5000$ image; accept $512 \times 512$ image.
   - Test payload limits: Reject file with byte length $> 15\text{ MB}$.

2. **Preprocessing Pipeline Tests (`tests/test_preprocess.py`):**
   - Test EXIF rotation: A vertically-flagged portrait image (EXIF orientation 6) is physically transposed to vertical dimensions.
   - Test channel conversion: Grayscale image (mode 'L') output tensor has shape `[1, 3, 224, 224]`.
   - Test transparency: RGBA image is properly composited onto white canvas without throwing dimension errors.
   - Test tensor values: Preprocessed tensor values fall within expected normalized bounds ($[-2.5, +2.5]$).

3. **Inference & Decision Engine Tests (`tests/test_inference.py`):**
   - Test `predict_image()` returns structured dict containing `predicted_label`, `model_score`, `calibrated_score`, `inference_time_ms`, and `abstention_flag`.
   - Test abstention threshold: A synthetic logit corresponding to score $0.50$ produces `predicted_label = "Inconclusive / Abstain"`.
   - Test determinism: Running inference twice on the identical image tensor yields identical score to 6 decimal places.

### 12.2 Academic Defense Acceptance Rubric

| Criterion | Failing Standard (Unacceptable) | Minimum Passing Standard | Distinction / Academic Excellence |
| :--- | :--- | :--- | :--- |
| **Model Work** | No training; only static heuristics or random mock | Fine-tuned lightweight CNN backbone on verified subset | Two-stage fine-tuning with temperature calibration & ablations |
| **Evaluation** | Only training set reported; claims 99% accuracy | Locked held-out test split evaluated; Confusion matrix reported | Full metric suite (ROC-AUC, ECE, F1) + External cross-generator evaluation |
| **Dataset Governance** | Unclear data origin; mixed/leaked test splits | Documented dataset source; explicit train/val/test splits | Cryptographic manifest with perceptual hash deduplication |
| **Honesty & Ethics** | Claims model "proves identity" or "detects all fakes" | Acknowledges limitations; provides basic disclaimer | Explicit abstention engine; error analysis on edge cases |
| **Reproducibility** | Broken code; hard-coded absolute paths; crashes | Runs locally after manual dependency troubleshooting | Single-command deterministic reproduction from seed to evaluation |

---

## 13. Appendix: Technical Reference, Configuration Schemas & APIs

### 13.1 Preprocessing Configuration Schema (`configs/preprocess.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "model_architecture": "efficientnet_b0",
  "checkpoint_path": "models/best_model.pth",
  "input_resolution": [224, 224],
  "interpolation": "bilinear",
  "antialias": true,
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "calibration": {
    "temperature": 1.1842,
    "method": "platt_scaling_validation"
  },
  "decision_thresholds": {
    "real_upper_bound": 0.35,
    "fake_lower_bound": 0.65,
    "abstention_enabled": true
  },
  "class_mapping": {
    "0": "Real / Authentic Consistent",
    "1": "AI-Generated / Synthetic"
  },
  "security_limits": {
    "max_file_size_bytes": 15728640,
    "min_dimension_px": 64,
    "max_dimension_px": 4096,
    "max_image_pixels": 16777216,
    "blur_variance_threshold": 100.0
  }
}
```

### 13.2 Model Card Specification (`reports/model_card.md`)

```markdown
# Model Card: EfficientNet-B0 Forensic Deepfake Detector

## Model Details
- **Architecture:** EfficientNet-B0 (Pretrained on ImageNet-1k, fine-tuned on GenImage subset)
- **Framework:** PyTorch 2.2+ / Torchvision 0.17+
- **Task:** Binary still-image AI-generation detection
- **Input:** Single static image tensor [1, 3, 224, 224], RGB
- **Output:** Calibrated synthetic probability score $s \in [0.0, 1.0]$ and 3-tier verdict

## Intended Use
- **Primary Use:** Academic research prototype and diagnostic triage tool for digital identity verification analysts.
- **Out-of-Scope:** Standalone automated identity denial, biometric facial recognition, video frame temporal analysis, presentation attack / liveness proof.

## Training Data & Methodology
- **Source:** GenImage Benchmark (Stable Diffusion v1.4 / v1.5 vs ImageNet-val authentic pairs)
- **Split Distribution:** 70% Train (7,000 samples), 15% Validation (1,500 samples), 15% Locked Test (1,500 samples)
- **Balancing:** 50% Real, 50% Synthetic
- **Optimizer:** AdamW (Weight Decay 1e-4), Cosine Annealing Learning Rate

## Evaluation & Performance Summary
- **Test Accuracy:** Reported in `reports/metrics.json`
- **ROC-AUC:** Reported in `reports/metrics.json`
- **Macro-F1:** Reported in `reports/metrics.json`
- **Expected Calibration Error (ECE):** Reported in `reports/metrics.json`

## Known Limitations & Failure Modes
1. **Generalization Gap:** Accuracy degrades when encountering unseen generative models (e.g., Midjourney v6, Flux) that utilize different latent decoders.
2. **Compression Vulnerability:** Aggressive lossy JPEG compression (Quality < 65) suppresses high-frequency generative traces, leading to abstention or false negatives.
3. **Facial Retouching Ambiguity:** Professional photography with commercial digital skin airbrushing may trigger elevated synthetic scores.
```

### 13.3 Python Environment Specification (`requirements.txt`)

```
# Core Machine Learning & Scientific Computing
torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121; sys_platform != 'darwin'
torchvision==0.17.2 --extra-index-url https://download.pytorch.org/whl/cu121; sys_platform != 'darwin'
torch==2.2.2; sys_platform == 'darwin'
torchvision==0.17.2; sys_platform == 'darwin'
numpy==1.26.4
scipy==1.12.0
scikit-learn==1.4.1.post1
pandas==2.2.1

# Image Processing, Forensics & Security
Pillow==10.2.0
imagehash==4.3.1
piexif==1.1.3
opencv-python-headless==4.9.0.80

# Visualization & Reporting
matplotlib==3.8.3
seaborn==0.13.2

# Application & UI Framework
streamlit==1.32.0

# Testing, Quality Assurance & Utilities
pytest==8.1.1
tqdm==4.66.2
```

### 13.4 Developer CLI Command Reference

```bash
# 1. Environment Setup
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt

# 2. Data Preparation & Manifest Generation
python src/dataset.py --action create_manifest --raw_dir data/raw --output_dir manifests/

# 3. Model Training
python src/train.py --config configs/train_config.json --preprocess configs/preprocess.json

# 4. Temperature Calibration
python src/calibration.py --val_manifest manifests/val_manifest.json --weights models/best_model.pth

# 5. Locked Test Evaluation & Reporting
python src/evaluate.py --test_manifest manifests/test_manifest.json --weights models/best_model.pth

# 6. Cross-Generator Generalization Evaluation
python src/evaluate.py --test_manifest manifests/external_manifest.json --weights models/best_model.pth --output_prefix generalization_

# 7. Launch Interactive Forensic Application
streamlit run app.py
```

---

## 14. Document Approval & Governance Sign-Off

| Review Role | Designated Authority | Status | Date |
| :--- | :--- | :--- | :--- |
| **Lead Machine Learning Researcher** | Antigravity AI Systems Team | Approved | 2026-11-17 |
| **Forensics & Computer Vision Lead** | Evaluation & Integrity Office | Approved | 2026-11-17 |
| **Identity Verification Domain Specialist**| Digital Trust & Safety Committee | Approved | 2026-11-17 |
| **Security & Privacy Officer** | Ethical AI & Data Governance Board | Approved | 2026-11-17 |

*This Product Requirements Document represents the binding engineering and scientific specification for the project. Any architectural deviations from this specification must be documented via formal revision control.*
