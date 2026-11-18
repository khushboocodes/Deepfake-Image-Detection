# Model Training, Testing & Forensic Telemetry Specification

## AI-Based Deepfake Image Detection System for Digital Identity Verification
**Document Version:** 2.0.0 (Production 10,000-Image Benchmark Specification)  
**Associated Architecture:** [`docs/architecture.md`](./architecture.md)  
**Associated Datasets Spec:** [`docs/datasets.md`](./datasets.md)  
**Associated Tech Stack:** [`docs/tech_stack.md`](./tech_stack.md)  
**Trained Model Checkpoint:** `models/best_model.pth`  
**Evaluation Metrics Report:** `reports/metrics.json`  

---

## 1. Executive Summary & Training Milestones

The AI-Based Deepfake Image Detection model was trained on an expanded, balanced benchmark of **10,000 high-resolution portrait images** derived from the Kaggle 140k Real & Fake Faces dataset (StyleGAN2 generations vs. authentic human face captures).

### Key Performance Summary (Locked 1,500-Image Test Split):
* **Test Accuracy:** **96.07%** (1,441 / 1,500 correct classifications)
* **ROC-AUC:** **0.9922** (Area Under the Receiver Operating Characteristic Curve)
* **PR-AUC:** **0.9929** (Precision-Recall Area Under Curve)
* **Synthetic Sensitivity / Recall:** **96.93%** (727 out of 750 fake images correctly flagged)
* **Real Specificity / TNR:** **95.20%** (714 out of 750 authentic images verified real)
* **False Alarm Rate (FPR):** **4.80%** (Low risk of falsely accusing real users)
* **Expected Calibration Error (ECE):** **0.0118** (Near-perfect probabilistic reliability)
* **GPU Inference Latency:** **~140 ms** on NVIDIA GeForce RTX 3050 Laptop GPU

---

## 2. Dataset Composition & Partitioning

### 2.1 Raw Benchmark Ingestion
The raw dataset was ingested from the Kaggle "140k Real and Fake Faces" benchmark (`archive (6)`), containing $50,000$ authentic human portraits from Flickr-Faces-HQ (FFHQ) and $50,000$ high-fidelity synthetic faces synthesized by StyleGAN2.

* **Raw Real Directory:** `data/raw/new_real/` ($50,000$ images)
* **Raw Synthetic Directory:** `data/raw/new_fake_stylegan/` ($50,000$ images)

### 2.2 Deduplication & Curation Protocol
To guarantee that identical images or near-duplicate crops do not contaminate both the training and evaluation splits:
1. **Perceptual Hashing (pHash):** Evaluated via 64-bit 2D Discrete Cosine Transform (DCT) on 32x32 luminance arrays:
   $$\text{Hamming Distance}(h_1, h_2) < 4 \implies \text{Deduplicated / Rejected}$$
2. **Balanced Sampling:** Exactly 5,000 Real and 5,000 Fake images were sampled and cataloged in the canonical dataset manifest (`manifests/manifest.json`).

### 2.3 Partitioning Splits (Deterministic Seed = 42)

```
┌────────────────────────────────────────────────────────────────────────┐
│                    10,000 BALANCED DATASET SPLITS                      │
├────────────────────┬──────────────┬──────────────┬─────────────────────┤
│ Split              │ Real Images  │ Fake Images  │ Total Images        │
├────────────────────┼──────────────┼──────────────┼─────────────────────┤
│ Training (70%)     │ 3,500        │ 3,500        │ 7,000               │
│ Validation (15%)   │ 750          │ 750          │ 1,500               │
│ Held-Out Test (15%)│ 750          │ 750          │ 1,500               │
├────────────────────┼──────────────┼──────────────┼─────────────────────┤
│ Total Cumulative   │ 5,000        │ 5,000        │ 10,000              │
└────────────────────┴──────────────┴──────────────┴─────────────────────┘
```

---

## 3. Hardware & Compute Environment

* **Target GPU:** Dedicated NVIDIA GeForce RTX 3050 Laptop GPU
* **VRAM:** 6,144 MB (6 GB GDDR6)
* **CUDA Driver / Runtime:** CUDA 12.4
* **cuDNN Acceleration:** Enabled (`torch.backends.cudnn.benchmark = True`)
* **Python Virtual Environment:** Python 3.10.11 (`.\.venv\Scripts\python.exe`)
* **PyTorch Version:** `2.6.0+cu124` with Torchvision `0.21.0+cu124`

---

## 4. Training Hyperparameters & Staged Strategy

Training a deep forensic classifier requires preserving high-frequency image textures (skin pores, eyelash boundaries, resampling checkerboard patterns) while preventing overfitting to background noise. We employed a **Two-Stage Fine-Tuning Protocol** configured in `configs/train_config.json`:

```
┌────────────────────────────────────────────────────────────────────────┐
│                     TWO-STAGE TRAINING REGIME                          │
├────────────────────────────┬───────────────────────────────────────────┤
│ Hyperparameter             │ Value & Configuration                     │
├────────────────────────────┼───────────────────────────────────────────┤
│ Stage 1 (Warmup) Epochs    │ 3 Epochs (Backbone frozen, Head trained)  │
│ Stage 1 Learning Rate      │ 5e-4 (0.0005)                             │
│ Stage 2 (Fine-Tune) Epochs │ 18 Epochs (Selective unfreezing Stages 6-7│
│ Stage 2 Backbone LR        │ 4e-5 (0.00004)                            │
│ Stage 2 Head LR            │ 3e-4 (0.0003)                             │
│ Batch Size                 │ 64 images per mini-batch                  │
│ Optimizer                  │ AdamW (betas=(0.9, 0.999), eps=1e-8)      │
│ Weight Decay (L2 Reg)      │ 1e-4 (0.0001)                             │
│ Loss Criterion             │ BCEWithLogitsLoss (pos_weight = 0.95)     │
│ Label Smoothing            │ 0.03 (prevents logit explosion)           │
│ LR Scheduler               │ CosineAnnealingLR (T_max = 18, eta=1e-6)  │
│ Early Stopping Patience    │ 6 epochs monitoring Validation Macro-F1   │
└────────────────────────────┴───────────────────────────────────────────┘
```

### Staged Training Rationale:
1. **Stage 1 (Head Warmup):** Prevents large random gradient updates from destroying pretrained ImageNet feature extractors in the convolutional stem.
2. **Stage 2 (Differential Learning Rates):** Fine-tunes high-level semantic representations (Stage 6, Stage 7 MBConv blocks) at a very gentle learning rate ($4 \times 10^{-5}$) while training the classification head at $3 \times 10^{-4}$.

---

## 5. Training Convergence & Accuracy Progress

```
Epoch 01/21 [Warmup]    - Loss: 0.4124 | Train Acc: 82.1% | Val Macro-F1: 0.8640 | Val Acc: 86.5%
Epoch 02/21 [Warmup]    - Loss: 0.3312 | Train Acc: 86.4% | Val Macro-F1: 0.8870 | Val Acc: 88.8%
Epoch 03/21 [Warmup]    - Loss: 0.2845 | Train Acc: 88.5% | Val Macro-F1: 0.9020 | Val Acc: 90.3%
--------------------------------------------------------------------------------------------------
Epoch 04/21 [Fine-Tune] - Loss: 0.2310 | Train Acc: 90.8% | Val Macro-F1: 0.9240 | Val Acc: 92.5%
Epoch 08/21 [Fine-Tune] - Loss: 0.1820 | Train Acc: 92.9% | Val Macro-F1: 0.9410 | Val Acc: 94.2%
Epoch 14/21 [Fine-Tune] - Loss: 0.1465 | Train Acc: 94.1% | Val Macro-F1: 0.9520 | Val Acc: 95.3%
Epoch 21/21 [Final]     - Loss: 0.1340 | Train Acc: 94.6% | Val Macro-F1: 0.9580 | Val Acc: 95.8%
```

* **Best Validation Macro-F1:** **0.9580** (Checkpoint saved to `models/best_model.pth`).
* **Final Training Accuracy:** **94.6%** with zero sign of catastrophic overfitting.

---

## 6. Comprehensive Test Set Evaluation (1,500 Samples)

The final checkpoint was evaluated on the locked, unread 1,500 test images using `src/evaluate.py`. Results are logged in `reports/metrics.json`:

### 6.1 Confusion Matrix

```
                      PREDICTED REAL        PREDICTED SYNTHETIC
ACTUAL REAL (750)     714 (TN = 95.2%)      36  (FP = 4.8%)
ACTUAL SYNTHETIC (750) 23  (FN = 3.1%)       727 (TP = 96.9%)
```

### 6.2 Key Statistical Metrics

| Metric | Scientific Definition / Formula | Value | Forensic Significance |
| :--- | :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | **96.07%** | Global overall correctness on balanced test set |
| **ROC-AUC** | Area under TPR vs FPR curve | **0.9922** | Near-perfect class separability across all thresholds |
| **PR-AUC** | Area under Precision vs Recall curve | **0.9929** | High precision maintained across full recall range |
| **Fake Recall / Sensitivity**| $\frac{TP}{TP + FN}$ | **96.93%** | Deepfake detection rate (only 23 fakes out of 750 missed) |
| **Real Specificity** | $\frac{TN}{TN + FP}$ | **95.20%** | Authentic human pass rate (714 out of 750 verified) |
| **False Alarm Rate (FPR)** | $\frac{FP}{FP + TN} = 1 - \text{Specificity}$ | **4.80%** | Minimizes false accusations against legitimate users |
| **Macro F1-Score** | $\frac{F1_{\text{real}} + F1_{\text{synthetic}}}{2}$ | **0.9607** | Balanced performance across both real and fake classes |
| **Equal Error Rate (EER)** | Point where $\text{FPR} = \text{FNR}$ | **3.87%** | Standard biometrics benchmark threshold |
| **Expected Calibration Error (ECE)** | $\sum_{m=1}^M \frac{\|B_m\|}{N} \|\text{acc}(B_m) - \text{conf}(B_m)\|$ | **0.0118** | Output probabilities match empirical reality ($< 1.2\%$ error) |
| **Brier Score** | $\frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$ | **0.0320** | Mean squared probability error (optimal = 0.0) |

---

## 7. Deep Dive: Dashboard Metrics & Telemetry Explained

Every metric shown on the Streamlit dashboard (`http://localhost:8501`) has a rigorous mathematical and forensic purpose:

```
┌────────────────────────────────────────────────────────────────────────┐
│                  DASHBOARD TELEMETRY REFERENCE GUIDE                   │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Calibrated Synthetic Likelihood (%)
* **What it is:** The primary output percentage (0.0% to 100.0%) indicating the estimated probability that the image contains generative AI / deepfake traces.
* **Formula:**
  $$s = \sigma\left(\frac{z}{T}\right) = \frac{1}{1 + \exp\left(-\frac{z}{0.9041}\right)} \times 100\%$$
* **How to interpret it:**
  * **Low Scores ($0\% - 20\%$):** Strong optical capture indicators. Skin pore regularity, natural sensor chromatic aberration, and authentic ocular reflections are detected. (e.g., $5.2\%$ synthetic means **$94.8\%$ authentic confidence**).
  * **Mid Scores ($35\% - 65\%$):** Ambiguous territory. The image may be heavily compressed, downscaled, or contains mixed signals. Mandates human analyst review in 3-Tier mode.
  * **High Scores ($80\% - 100\%$):** Strong generative neural network traces. Resampling checkerboard artifacts, asymmetric ear cartilage, or synthetic iris blending detected.

### 2. Neural Network Raw Logit ($z$)
* **What it is:** The unnormalized real-valued scalar output from the final linear layer of EfficientNet-B0 before any sigmoid or temperature scaling is applied.
* **Range:** $(-\infty, +\infty)$
* **How to interpret it:**
  * **Negative Logits ($z < 0$, e.g. $-4.53$):** Pushes prediction towards Real. The more negative the number, the higher the model's confidence that the photo is authentic.
  * **Zero Logit ($z = 0.0$):** Exactly 50/50 uncertainty boundary.
  * **Positive Logits ($z > 0$, e.g. $+2.28$):** Pushes prediction towards Synthetic. The more positive the number, the stronger the deepfake signals detected.

### 3. Model Latency (ms) vs. Total Pipeline Time (ms)
* **Model Latency:** Time taken by the **NVIDIA GeForce RTX 3050 GPU** to execute the forward passes for both the Global Composition View and Central Face ROI Zoom View. Typically **$130 - 180\text{ ms}$**.
* **Total Pipeline Time:** The complete wall-clock time from byte ingestion to UI rendering. Typically **$180 - 220\text{ ms}$**.
  * Pipeline breakdown: Magic byte verification ($5\text{ ms}$) + EXIF extraction ($8\text{ ms}$) + Laplacian focus calculation ($15\text{ ms}$) + Dual-scale tensor transforms ($20\text{ ms}$) + GPU inference ($140\text{ ms}$) + Post-processing & calibration ($2\text{ ms}$).

### 4. SHA-256 Digest
* **What it is:** A 256-bit (64 hexadecimal character) cryptographic hash generated from the raw image byte stream.
* **Why it is shown:** In digital forensics, law enforcement, and KYC audits, the SHA-256 hash serves as an **immutable digital fingerprint**. It guarantees the "Chain of Custody" — proving that the exact image evaluated by the AI model has not been altered, cropped, or tampered with before or after the inspection.

### 5. Focus Laplacian Variance (Threshold: $\ge 100.0$)
* **What it is:** A computer vision metric that measures the sharpness and high-frequency edge density of the image.
* **Formula:**
  $$\sigma_{\text{Laplace}}^2 = \text{Variance}\left( \nabla^2 I \right)$$
* **Why it is shown:** Generative AI models often leave subtle high-frequency artifacts in skin pores and eyes. If an image is blurry, out-of-focus, or aggressively compressed, those high-frequency clues disappear.
  * **$\ge 100.0$ (Normal/Sharp):** Image has sufficient micro-texture for high-confidence forensic analysis.
  * **$< 100.0$ (Blur / Soft Focus Warning):** System issues an advisory banner because blur softens the high-frequency camera sensor noise.

### 6. Shannon Entropy (bits/pixel)
* **What it is:** An information theory metric measuring the average information density and randomness across color channels.
* **Formula:**
  $$H(I) = -\sum_{k=0}^{255} p(k) \log_2 p(k)$$
* **Why it is shown:** Flat, solid, or artificially smoothed images have low entropy ($< 6.0\text{ bits/pixel}$), while rich natural human portraits with detailed hair, skin texture, and backgrounds score between **$7.0$ and $7.8\text{ bits/pixel}$**.

### 7. EXIF Metadata (Camera Make, Model, Software, Timestamp)
* **What it is:** Header tags embedded by camera hardware (e.g. Apple iPhone, Sony, Nikon) at the moment of physical capture.
* **Why it is shown:** Helps detect editing software signatures (e.g. "Adobe Photoshop", "StableDiffusion Inpaint").
* **Why Web Images Show "None":** Major messaging apps (WhatsApp, Telegram) and social platforms (Twitter, Instagram) automatically strip EXIF metadata to protect user privacy. Therefore, `None` is completely normal and is never used alone to classify an image as fake.

### 8. Temperature $T$ ($0.9041$)
* **What it is:** The scaling factor derived from Platt Temperature Scaling on validation logits.
* **Why it is shown:** Deep neural networks are known to output uncalibrated probabilities (often overconfident near 0.0 or 1.0). Scaling logits by $T = 0.9041$ ensures that a predicted score of $90\%$ means that out of 100 such images, exactly 90 are empirically synthetic.

### 9. Decision Rules: Direct Mode vs. Forensic 3-Tier Mode
* **Direct Verdict Mode ($\tau = 0.58$):**
  * Binary threshold tuned on ROC curve: Score $< 0.58 \implies \text{Real}$, Score $\ge 0.58 \implies \text{Synthetic}$.
  * Best for rapid screening with $96.07\%$ accuracy and minimal false alarms.
* **Forensic 3-Tier Mode (Legal & KYC Triage):**
  * Introduces an explicit safety buffer:
    * $\le 35\%$: Verified Authentic
    * $\ge 65\%$: Flagged Synthetic
    * $35\% - 65\%$: Inconclusive (Abstain — mandates review by a human forensic specialist).

---

## 8. Summary of Retraining & Improvement History

```
┌────────────────────────────────────────────────────────────────────────┐
│                     EVOLUTION OF MODEL ACCURACY                        │
├────────────────────┬──────────────┬──────────────┬─────────────────────┤
│ Metric             │ Baseline     │ Option A     │ Production (10k)    │
│                    │ (1,400 img)  │ (1,400 tuned)│ (10,000 images)     │
├────────────────────┼──────────────┼──────────────┼─────────────────────┤
│ Test Accuracy      │ 71.20%       │ 84.50%       │ 96.07% (+24.87%)    │
│ ROC-AUC            │ 0.7850       │ 0.9120       │ 0.9922 (+0.2072)    │
│ Real Specificity   │ 68.40%       │ 82.10%       │ 95.20% (+26.80%)    │
│ Fake Sensitivity   │ 74.00%       │ 86.90%       │ 96.93% (+22.93%)    │
│ False Alarm Rate   │ 31.60%       │ 17.90%       │ 4.80% (-26.80%)     │
│ Calibration ECE    │ 0.1420       │ 0.0540       │ 0.0118 (-0.1302)    │
│ Decision Mode      │ Binary 0.50  │ Binary 0.58  │ Dual (Direct/3-Tier)│
└────────────────────┴──────────────┴──────────────┴─────────────────────┘
```

*This document serves as the permanent scientific and operational reference for model training, validation, testing, and telemetry interpretation.*
