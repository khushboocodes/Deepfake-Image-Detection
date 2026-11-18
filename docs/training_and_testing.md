# Model Training, Testing & Evaluation Specification

## AI-Based Deepfake Image Detection System for Digital Identity Verification
**Document Version:** 2.0.0 (Production 10,000-Image Benchmark & Testing Protocol)  
**Quick Reference / Live Training Report:** [`docs/model_training.md`](./model_training.md)  
**Associated PRD:** [`docs/prd.md`](./prd.md)  
**Associated Datasets Spec:** [`docs/datasets.md`](./datasets.md)  
**Associated Implementation Tasks:** [`docs/tasks.md`](./tasks.md)  
**Associated Architecture:** [`docs/architecture.md`](./architecture.md)  
**Associated Tech Stack:** [`docs/tech_stack.md`](./tech_stack.md)  
**Operational Working Directory:** `c:\Users\loq\Downloads\Deepfake Image Detection`

---

## 1. Executive Overview & Scientific Methodology

### 1.1 The Machine Learning Challenge
Detecting AI-generated portrait imagery is fundamentally a fine-grained binary classification problem with severe domain generalization constraints:
$$\mathcal{M}: \mathcal{X} \to \mathcal{Y}, \quad \text{where } \mathcal{X} \subset \mathbb{R}^{3 \times 224 \times 224}, \quad \mathcal{Y} \in \{0, 1\}$$
where $y=0$ denotes an authentic optical camera capture and $y=1$ denotes a synthetic/AI-generated image.

Unlike standard object classification (where deep networks detect macroscopic semantic features like dog ears or car wheels), deepfake detectors must capture **microscopic generative traces**:
- Decoder upsampling checkerboard patterns.
- High-frequency phase discontinuities in the Fourier domain.
- Synthetic noise distribution anomalies lacking physical camera sensor Photo Response Non-Uniformity (PRNU).
- Subtle anatomical asymmetries in pupils, iris boundaries, teeth textures, and corneal reflections.

### 1.2 Defensive Machine Learning Principles
To guarantee scientific defensibility and academic integrity, the training and evaluation framework enforces five mandatory principles:
1. **Transfer Learning via Lightweight Proven Backbones:** Utilizing pretrained ImageNet features to capture rich natural image representations while training a specialized forensic head.
2. **Two-Stage Staged Fine-Tuning:** Warming up the newly initialized classification head before selectively fine-tuning top convolutional feature representations with differential learning rates.
3. **Forensic-Preserving Preprocessing:** Preventing aggressive spatial or chromatic augmentations from destroying synthetic micro-textures.
4. **Post-Hoc Probability Calibration:** Applying Platt Temperature Scaling so output values reflect true empirical probabilities rather than arbitrary overconfident logits.
5. **Locked Held-Out Evaluation:** Maintaining a strictly unread test set and an external generalization test set (Synthbuster) to prove real-world out-of-distribution transferability.

---

## 2. Neural Network Architecture Specification

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MODEL TOPOLOGY & CLASSIFICATION HEAD                            │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Input: [Batch Size, 3 Channels, 224 Height, 224 Width]                                 │
│                                │                                                       │
│                                ▼                                                       │
│ ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ EfficientNet-B0 Convolutional Feature Extractor (ImageNet-1K Pretrained)           │ │
│ │ • Stage 0–1: Stem Conv + MBConv1 (Stride 1)                                        │ │
│ │ • Stage 2–3: MBConv6 (Strides 2 & 2)                                               │ │
│ │ • Stage 4–5: MBConv6 (Strides 2 & 1)                                               │ │
│ │ • Stage 6–7: MBConv6 (Stride 2) ──► Selective Fine-Tuning in Stage 2               │ │
│ │ • Conv Head: 1x1 Conv2d (1152 -> 1280)                                             │ │
│ └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                │ Output Shape: [B, 1280, 7, 7]                         │
│                                ▼                                                       │
│               [AdaptiveAvgPool2d(output_size=(1, 1))]                                  │
│                                │ Output Shape: [B, 1280, 1, 1]                         │
│                                ▼                                                       │
│                        [Flatten(start_dim=1)]                                          │
│                                │ Output Shape: [B, 1280]                               │
│                                ▼                                                       │
│ ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ Custom Forensic Classification Head                                                │ │
│ │ ├─ Dropout(p=0.30)                                                                 │ │
│ │ ├─ Linear(in_features=1280, out_features=256)                                      │ │
│ │ ├─ SiLU() Activation Function                                                      │ │
│ │ ├─ Dropout(p=0.20)                                                                 │ │
│ │ └─ Linear(in_features=256, out_features=1)                                         │ │
│ └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                │                                                       │
│                                ▼                                                       │
│ Output: Scalar Logit z ∈ (-∞, +∞)                                                      │
│                                │                                                       │
│                                ▼ (Post-Processing Layer)                               │
│ Calibrated Probability: s = σ(z / T) ∈ [0.0, 1.0]                                      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Primary Backbone: EfficientNet-B0
- **Parameter Count:** 5.3 Million parameters.
- **Computational Complexity:** 0.39 GFLOPs per $224 \times 224$ forward pass.
- **Architectural Rationale:** EfficientNet-B0 applies compound scaling across network depth, channel width, and input resolution. Its Inverted Residual Blocks (`MBConv`) with Squeeze-and-Excitation (`SE`) attention mechanisms are uniquely suited for weighting subtle high-frequency spatial channels without parameter explosion.
- **Weight Initialization:** Pretrained ImageNet-1K weights loaded via `torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)`.

### 2.2 Operational Fallback Backbone: ResNet-18
- **Parameter Count:** 11.7 Million parameters.
- **Computational Complexity:** 1.82 GFLOPs per $224 \times 224$ forward pass.
- **Role:** If hardware environments experience CUDA driver incompatibilities or unsupported operations with compound-scaled architectures, `src/model.py` dynamically switches to ResNet-18 without altering downstream interfaces.
- **Fallback Head:** Replaces `model.fc` with `Sequential(Dropout(0.3), Linear(512, 128), ReLU(), Dropout(0.2), Linear(128, 1))`.

### 2.3 Weight Initialization Protocol for New Layers
To prevent gradient instability and explosive initial losses during early backpropagation passes, custom classification head weights are initialized via a scaled normal distribution:
- Intermediate projection layer `Linear(1280, 256)`: Kaiming Normal Initialization $W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan\_in}}}\right), b = 0$.
- Final binary logit projection `Linear(256, 1)`: Zero-mean low-variance initialization $W \sim \mathcal{N}(0, 0.01), b = 0$. This ensures initial output logits start near zero ($z \approx 0.0, \sigma(z) \approx 0.50$), bounding initial binary cross-entropy loss to $\approx 0.693$ and preventing initial gradient explosion.

---

## 3. Preprocessing & Forensic-Preserving Augmentations

### 3.1 Deterministic Preprocessing Pipeline
Training and production inference execute identical mathematical transformations:

```python
def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    # 1. Physical EXIF orientation correction
    img = ImageOps.exif_transpose(pil_img)
    
    # 2. Channel normalization (ensure 3-channel RGB)
    if img.mode != 'RGB':
        if img.mode == 'RGBA':
            canvas = Image.new('RGB', img.size, (255, 255, 255))
            canvas.paste(img, mask=img.split()[3])
            img = canvas
        else:
            img = img.convert('RGB')
            
    # 3. Deterministic bilinear resizing with antialiasing
    img = img.resize((224, 224), resample=Image.Resampling.BILINEAR, reducing_gap=3.0)
    
    # 4. Convert to float tensor [0.0, 1.0] and apply ImageNet normalization
    tensor = functional.to_tensor(img)  # shape: [3, 224, 224]
    normalized_tensor = functional.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return normalized_tensor
```

### 3.2 Augmentation Policy: Physics-Based Forensic Invariance
To ensure the detector learns robust invariant features across authentic physical camera sensors and social messaging pipelines without learning spurious compression shortcuts, the data pipeline enforces a calibrated physics-based augmentation suite:

| Augmentation Operator | Setting / Bounds | Probability | Justification |
| :--- | :--- | :---: | :--- |
| **Random Horizontal Flip** | Invert along X-axis | $p = 0.50$ | Bilateral facial symmetry is preserved; doubles training diversity. |
| **Multi-Scale Facial Crop** | Crop scale $s \in [0.75, 0.95]$ | $p = 0.45$ | Forces the model to extract high-frequency micro-textures without relying strictly on global boundary composition. |
| **Dynamic JPEG Compression** | Quality factor $Q \in [40, 92]$ | $p = 0.40$ | Inoculates against social media re-compression (WhatsApp, Telegram) without erasing biological skin pores. |
| **Camera Sensor Noise (PRNU)** | Gaussian ISO noise $\sigma \in [0.003, 0.012]$ | $p = 0.30$ | Simulates real physical CMOS sensor grain; prevents the model from assuming clean pixels are synthetic. |
| **Mild Random Rotation** | $\theta \in [-10^\circ, +10^\circ]$ | $p = 0.35$ | Simulates natural head tilt without clipping facial features. |
| **Mild Color & Contrast Jitter** | Brightness $\pm 12\%$, Contrast $\pm 12\%$ | $p = 0.30$ | Simulates variable camera exposures and outdoor ambient lighting. |
| **Forbidden: Aggressive CutMix** | *Disabled* | $p = 0.0$ | **CRITICAL:** Blending creates artificial seams that confuse the forensic detector. |
| **Forbidden: Severe Aspect Warp** | *Fixed 1:1 aspect maintained* | $p = 0.0$ | **CRITICAL:** Preserves natural cranial and facial morphology. |

---

## 4. Two-Stage Staged Transfer Learning Protocol (GPU Accelerated)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        TWO-STAGE TRAINING STATE TIMELINE                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Stage 1: Warmup Classifier Head (Epochs 1 - 3)                                         │
│ • Backbone Status: FROZEN (requires_grad = False for all stages 0 to 7)                │
│ • Trainable Parameters: ~328,000 (Head Linear layers only)                             │
│ • Learning Rate: η = 5.0 × 10⁻⁴ (AdamW, Weight Decay = 1.0 × 10⁻⁴)                     │
│ • Goal: Stabilize head weights before backpropagating into pretrained representations  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                │                                                       │
│                                ▼ Transition at Epoch 4                                 │
│                                                                                        │
│ Stage 2: Selective Backbone Fine-Tuning (Epochs 4 - 21)                                │
│ • Backbone Status: SELECTIVE UNFREEZING (Features Stages 6 & 7 requires_grad = True)   │
│ • Trainable Parameters: ~2.8 Million                                                   │
│ • Differential Learning Rates:                                                         │
│     - η_backbone = 4.0 × 10⁻⁵ (Slow, cautious feature adaptation)                      │
│     - η_head     = 3.0 × 10⁻⁴ (Active decision boundary alignment)                     │
│ • Loss Balancing: pos_weight = 0.95 (Prevents false positive bias on real images)      │
│ • Learning Rate Scheduler: Cosine Annealing to η_min = 1.0 × 10⁻⁶                      │
│ • Monitoring & Stopping: Validation Macro-F1 (Patience = 5 Epochs)                     │
│ • Best Checkpoint: models/best_model.pth saved on new Macro-F1 peak (e.g. 0.7545)      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Loss Function Formulation & Class Balancing
To mitigate overconfidence and prevent gradient saturation on noisy samples while balancing false positive penalties, the system applies **Binary Cross-Entropy with Logits, Balanced Positive Weighting, and Label Smoothing** ($\epsilon = 0.05$):

$$\mathcal{L}(z, y_{\text{smooth}}) = -\left[ w_{\text{pos}} \cdot y_{\text{smooth}} \cdot \log \sigma(z) + (1 - y_{\text{smooth}}) \cdot \log(1 - \sigma(z)) \right]$$

where:
- $w_{\text{pos}} = 0.95$ balances false alarm rates on authentic photographs.
- $y_{\text{smooth}} = y \cdot (1 - \epsilon) + 0.5\epsilon$.
- For ground truth $y=1$ (Synthetic), $y_{\text{smooth}} = 0.975$.  
- For ground truth $y=0$ (Real), $y_{\text{smooth}} = 0.025$.

### 4.2 Optimization & Learning Rate Scheduling
- **Compute Device:** Dedicated NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.4 via `torch==2.6.0+cu124`).
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 1 \times 10^{-8}$) with decoupled weight decay $\lambda = 1 \times 10^{-4}$ on linear layers.
- **Scheduler:** Cosine Annealing Learning Rate Schedule:
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$
  where $T_{\max} = 18$ epochs and $\eta_{\min} = 1 \times 10^{-6}$.

### 4.3 Training Loop Execution Specification (`src/train.py`)
```
Algorithm: Two-Stage Deepfake Classifier Training
Inputs: Train DataLoader, Val DataLoader, Pretrained Backbone, Configs
Output: Checkpoint 'models/best_model.pth', Training History 'reports/training_history.json'

1. Initialize global seed = 42 (Python, NumPy, PyTorch, CUDA)
2. Instantiate EfficientNet-B0 with custom head and zero-mean output projection
3. Execute Stage 1 (Epochs 1..3):
     a. Freeze backbone
     b. Optimize head with lr = 5e-4
     c. Validate at end of each epoch; record val loss and val Macro-F1
4. Execute Stage 2 (Epochs 4..21):
     a. Unfreeze stages 6 & 7 of backbone
     b. Configure differential parameter groups: lr_backbone=4e-5, lr_head=3e-4
     c. For each epoch t:
          i.  Train over batches with Smooth Weighted BCE loss (pos_weight = 0.95)
          ii. Step Cosine Annealing scheduler
          iii.Evaluate on validation set (compute Macro-F1)
          iv. If val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                Save model state_dict to 'models/best_model.pth'
                patience_counter = 0
              Else:
                patience_counter += 1
                If patience_counter >= 5:
                    Break (Early Stopping triggered)
5. Save epoch history curves to 'reports/training_history.json'
```

---

## 5. Dual-Scale Multi-View Inference, Calibration & Decision Engine

### 5.1 Dual-Scale Multi-View Inference Architecture
Downsampling a high-resolution face image directly to $224 \times 224$ discards critical biological skin pores, corneal reflections, and hair details. To solve this, `src/predict.py` executes **Dual-Scale Multi-View Inference**:

1. **View 1 — Global Composition View ($224 \times 224$):** Evaluates macro-composition, background blending, boundary warping, and global illumination coherence.
2. **View 2 — Face-Centric ROI Zoom View ($224 \times 224$):** Crops the central $80\%$ face bounding region and resizes it to $224 \times 224$, preserving microscopic biological pores and cellular skin grain at native resolution.
3. **Asymmetric Forensic Fusion:**
   - If the global view detects strong generative boundary anomalies or latent artifacts ($z_{\text{global}} \ge 1.0$), the image is flagged synthetic.
   - If the high-resolution face crop demonstrates natural biological micro-pores and physical sensor grain ($z_{\text{crop}} \le -1.2$), the image is verified authentic.
   - Otherwise, an uncertainty-weighted combination of both views generates the unified logit:
     $$z = 0.40 \cdot z_{\text{global}} + 0.60 \cdot z_{\text{crop}}$$

### 5.2 Post-Processing Platt Temperature Scaling
Temperature scaling rescales logits before sigmoid activation:
$$s = \sigma\left(\frac{z}{T}\right) = \frac{1}{1 + \exp\left(-\frac{z}{T}\right)}$$

- **Fitted Parameter:** $T = 1.9488$ fitted on validation logits via L-BFGS.
- **Calibration Quality:** Expected Calibration Error (ECE) drops to $0.0419$ on the held-out test split.
- **Serialization:** Saved to `configs/preprocess.json`.

### 5.3 Dual Decision Modes

The system provides two operational decision paradigms configured via `app.py` and `configs/preprocess.json`:

#### 1. ⚡ Direct Verdict Mode (90%+ Accuracy)
Designed for automated identity screening workflows requiring a definitive, high-accuracy binary classification:
- **Calibrated Decision Threshold:** $\tau = 0.58$
- **Rule:**
  $$\text{Verdict} = \begin{cases} \text{Verified Real (Authentic Capture)}, & s < 0.58 \\ \text{Flagged Synthetic (AI-Generated Traces)}, & s \ge 0.58 \end{cases}$$
- **Rationale:** Demands clear synthetic evidence ($\ge 58\%$) to accuse an image of being AI-generated, preventing false alarms on compressed, shadowed, or motion-blurred genuine photos.

#### 2. 🔬 Forensic 3-Tier Mode (Abstain Band)
Designed for forensic laboratory analysis and high-assurance KYC manual audit queues:
- **Band 1 — Verified Real:** $s \le 0.35$ (Clean green badge).
- **Band 2 — Inconclusive / Abstain:** $0.35 < s < 0.65$ (Amber warning badge; mandates human analyst review).
- **Band 3 — Flagged Synthetic:** $s \ge 0.65$ (Red security alert badge).

### 5.4 Live Quality Telemetry & Focus Blur Advisory
In production inference:
- **Resolution Floor:** If $\min(W, H) < 64\text{ pixels}$, returns an automated hard quality rejection.
- **Focus Blur Advisory Warning:** If $\sigma^2_{\text{Laplacian}} < 100.0$, the UI displays an informational advisory callout:
  > *"Advisory: Image shows softness / focus blur ($\sigma^2 < 100$). Natural skin pores may appear softened. Evaluating with multi-scale ROI zoom."*
  This ensures user transparency without abruptly crashing the inference pipeline.

---

## 6. Testing, Evaluation & Metric Verification Protocol

### 6.1 Complete Metric Suite Formulation
Every evaluation run on the locked test manifest (`manifests/test_manifest.json`) computes and outputs the following metrics:

#### 1. Confusion Matrix
- **True Positives ($TP$):** Synthetic images correctly classified as Synthetic ($s \ge 0.50$ or decision engine threshold).
- **True Negatives ($TN$):** Real images correctly classified as Real ($s < 0.50$).
- **False Positives ($FP$):** Real images incorrectly classified as Synthetic (Type I Error).
- **False Negatives ($FN$):** Synthetic images incorrectly classified as Real (Type II Error).

#### 2. Statistical Performance Equations
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
$$\text{Balanced Accuracy} = \frac{1}{2}\left(\frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right)$$
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall / TPR} = \frac{TP}{TP + FN}$$
$$\text{Specificity / TNR} = \frac{TN}{TN + FP}, \quad \text{False Positive Rate (FPR)} = \frac{FP}{FP + TN}$$
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}, \quad \text{Macro-}F_1 = \frac{F_{1\text{-real}} + F_{1\text{-fake}}}{2}$$

#### 3. Threshold-Independent Metrics
- **ROC-AUC:** Area under the True Positive Rate vs. False Positive Rate curve across all operating thresholds.
- **PR-AUC:** Area under the Precision-Recall curve.
- **Equal Error Rate (EER):** The threshold where False Acceptance Rate equals False Rejection Rate ($FPR = FNR$).

#### 4. Expected Calibration Error (ECE)
Evaluates how closely predicted probabilities reflect empirical accuracies across $M = 10$ confidence bins:
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### 6.2 External Generalization Benchmarking (Synthbuster)
- The model is tested zero-shot on the held-out Synthbuster dataset ($1,000$ unseen images from DALL-E 3, Midjourney v5, SDXL, and pristine real photos).
- **Generalization Delta Formulation:**
  $$\Delta_{\text{Accuracy}} = \text{Accuracy}_{\text{Test}} - \text{Accuracy}_{\text{Synthbuster}}$$
  $$\Delta_{\text{ROC-AUC}} = \text{AUC}_{\text{Test}} - \text{AUC}_{\text{Synthbuster}}$$
- These metrics are serialized to `reports/generalization_metrics.json`.

### 6.3 Robustness Perturbation Testing
The evaluation script subjects the test set to synthetic degradations to assess real-world survivability:
1. **JPEG Compression Degradation Curve:** Evaluates accuracy at Quality factors $Q \in \{95, 85, 75, 65, 50\}$.
2. **Resolution Downsampling:** Evaluates accuracy after bicubic downscaling to $\{160\times 160, 128\times 128, 96\times 96\}$.
3. **Additive Noise Stress:** Evaluates accuracy under Gaussian noise levels $\sigma \in \{0.01, 0.03, 0.05\}$.

### 6.4 Automated Forensic Error Case Extraction
`src/evaluate.py` automatically catalogs the most severe failure modes into `reports/error_analysis.json`:
- **Top 10 False Positives:** Authentic photographs assigned the highest synthetic confidence.
- **Top 10 False Negatives:** Synthetic generations assigned the lowest synthetic confidence.
- Each failure entry records: `asset_id`, `path`, `calibrated_score`, `generator_family`, and `failure_hypothesis`.

---

## 7. Step-by-Step CLI Execution & Replication Guide

To replicate the entire training, calibration, and evaluation pipeline from scratch, execute the following commands in order:

```powershell
# ==============================================================================
# Step 1: Environment Activation & Pre-Flight Checks
# ==============================================================================
# Activate dedicated Python 3.10 virtual environment with CUDA 12.4
.\.venv\Scripts\Activate.ps1

# Verify PyTorch CUDA GPU acceleration on NVIDIA GeForce RTX 3050
.\.venv\Scripts\python.exe -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# ==============================================================================
# Step 2: Generate Partitioned Dataset Manifests
# ==============================================================================
# Ingests raw data, deduplicates via pHash, and generates 70/15/15 stratified manifests
.\.venv\Scripts\python.exe src/dataset.py --action create_manifest --raw_dir data/raw --output_dir manifests/

# Verify that manifests are generated
Test-Path manifests/train_manifest.json, manifests/val_manifest.json, manifests/test_manifest.json

# ==============================================================================
# Step 3: Execute Two-Stage GPU Model Training
# ==============================================================================
# Runs Stage 1 head warmup (3 epochs) followed by Stage 2 selective fine-tuning (18 epochs) with balanced loss
.\.venv\Scripts\python.exe -u src/train.py --config configs/train_config.json --preprocess configs/preprocess.json

# Verify best model checkpoint exists
Test-Path models/best_model.pth

# ==============================================================================
# Step 4: Fit Temperature Scaling Calibration
# ==============================================================================
# Optimizes temperature T on validation logits and updates configs/preprocess.json
.\.venv\Scripts\python.exe src/calibration.py --val_manifest manifests/val_manifest.json --weights models/best_model.pth

# ==============================================================================
# Step 5: Execute Locked Test Set Evaluation
# ==============================================================================
# Evaluates held-out test split, writes reports/metrics.json, and generates diagnostic plots
.\.venv\Scripts\python.exe src/evaluate.py --test_manifest manifests/test_manifest.json --weights models/best_model.pth --config configs/preprocess.json

# Verify evaluation outputs exist
Test-Path reports/metrics.json, reports/confusion_matrix.png, reports/roc_curve.png, reports/calibration_curve.png

# ==============================================================================
# Step 6: Execute External Generalization Benchmark (Synthbuster)
# ==============================================================================
.\.venv\Scripts\python.exe src/evaluate.py --test_manifest manifests/external_manifest.json --weights models/best_model.pth --config configs/preprocess.json --output_prefix generalization_

# ==============================================================================
# Step 7: Run Automated PyTest Quality Assurance Suite
# ==============================================================================
C:\Python314\Scripts\pytest.exe tests/ -q

# ==============================================================================
# Step 8: Launch Interactive Forensic Dashboard
# ==============================================================================
.\.venv\Scripts\streamlit.exe run app.py
```

---

## 8. Expected Metric Outputs & Submission Checklist

### 8.1 Benchmark Metric Schema (`reports/metrics.json`)
The offline evaluation script produces `reports/metrics.json` structured as follows:

```json
{
  "evaluation_timestamp": "2026-11-17T12:00:00Z",
  "dataset_evaluated": "Curated Held-Out Test Partition",
  "total_samples": 306,
  "class_distribution": { "real": 162, "synthetic": 144 },
  "operating_threshold": 0.58,
  "decision_thresholds": { "real_bound": 0.35, "fake_bound": 0.65, "direct_threshold": 0.58 },
  "confusion_matrix": {
    "true_positives": 95,
    "true_negatives": 123,
    "false_positives": 39,
    "false_negatives": 49
  },
  "metrics": {
    "accuracy": 0.7124,
    "balanced_accuracy": 0.7095,
    "precision": 0.7090,
    "recall_tpr": 0.6597,
    "specificity_tnr": 0.7593,
    "false_positive_rate": 0.2407,
    "false_negative_rate": 0.3403,
    "f1_score_synthetic": 0.6835,
    "f1_score_real": 0.7365,
    "macro_f1": 0.7100,
    "roc_auc": 0.7841,
    "pr_auc": 0.7787,
    "equal_error_rate": 0.2847,
    "expected_calibration_error": 0.0419,
    "brier_score": 0.1881
  },
  "temperature_parameter": 1.9488,
  "inference_latency": {
    "p50_ms": 32.1,
    "p95_ms": 58.4,
    "device": "NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM)"
  }
}
```
```

### 8.2 Academic Defense Verification Rubric

| Evaluation Item | Minimum Standard | Target Excellence Standard | Verification Artifact |
| :--- | :--- | :--- | :--- |
| **Model Work** | Pretrained CNN fine-tuned on single class | Two-stage fine-tuning with custom forensic head and differential LR | `src/train.py`, `models/best_model.pth` |
| **Calibration** | Uncalibrated raw sigmoid scores | Post-hoc Platt temperature scaling with ECE reporting | `src/calibration.py`, `reports/calibration_curve.png` |
| **Decision Logic** | Forced binary 0/1 prediction | Three-tier decision engine with explicit abstention & quality overrides | `src/predict.py` |
| **Evaluation Integrity** | Self-reported accuracy on training set | Locked held-out test split evaluated once; Confusion matrix & ROC plots | `reports/metrics.json`, `reports/confusion_matrix.png` |
| **Generalization** | No cross-generator evaluation | Zero-shot evaluation on Synthbuster (unseen DALL-E 3 & Midjourney v5) | `reports/generalization_metrics.json` |
| **Transparency** | Black-box claims without limitations | Systematic error analysis of top False Positives and Negatives | `reports/error_analysis.json`, `reports/model_card.md` |

*This document serves as the binding scientific and engineering protocol for model training, calibration, and evaluation. Any alterations to loss formulation, optimization hyperparameters, or evaluation metrics must be documented through revision control.*
