# Model Card: EfficientNet-B0 Forensic Deepfake Detector

## Model Details
- **Architecture Name:** `efficientnet_b0` with custom forensic classification head
- **Base Backbone:** Inverted Residual Blocks (MBConv) with Squeeze-and-Excitation (SE) channel attention
- **Custom Classification Head:**
  `AdaptiveAvgPool2d(1x1) -> Flatten -> Dropout(0.35) -> Linear(1280, 256) -> BatchNorm1d(256) -> SiLU -> Dropout(0.2) -> Linear(256, 1)`
- **Head Weight Initialization:** Intermediate linear layer via Kaiming normal; final binary projection layer via $\mathcal{N}(0.0, 0.01)$ to ensure stable zero-mean initial logits ($z \approx 0.0, \text{Loss} \approx 0.693$) and eliminate explosive gradient oscillations.
- **Framework & Compute:** PyTorch 2.6.0+cu124 / Torchvision 0.21.0+cu124 on dedicated NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, CUDA 12.4).
- **Task:** Binary still-image AI-generation detection (Real / Authentic vs. Synthetic / AI-Generated).
- **Input Geometry:** Tensor `[B, 3, 224, 224]`, RGB mode, normalized with ImageNet constants ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Multi-View Inference Engine:** Dual-Scale Asymmetric Multi-View:
  - *View 1 (Global Composition):* Standard $224 \times 224$ view assessing macro-lighting, background blending, and boundary warping.
  - *View 2 (Face-Centric ROI Zoom):* Central $80\%$ facial crop resized to $224 \times 224$, preserving native biological skin pores, corneal reflections, and hair details.
  - *Asymmetric Fusion:* Flags synthetic if boundary/warping artifacts $\ge 1.0$; verifies real if facial micro-pores $\le -1.2$.
- **Output:** Calibrated synthetic probability score $s \in [0.0, 1.0]$, raw logit $z \in (-\infty, +\infty)$, and mode-dependent categorical verdict.

---

## Intended Use
- **Primary Use Case:** Academic research prototype and diagnostic triage assistant for digital identity and KYC risk analysts.
- **Workflow Role:** Flags suspicious portrait uploads for secondary human inspection; fast-tracks consistent natural captures.
- **Explicitly Prohibited Uses:**
  - Autonomous denial of financial accounts, credit, government identity verification, or employment.
  - Biometric 1:1 identity verification or 1:N facial matching.
  - Physical human liveness or presentation attack detection (PAD).
  - Standalone evidence in civil or criminal legal proceedings.

---

## Training Data & Methodology
- **Source Benchmarks:** Curated 10,000-sample balanced dataset derived from Kaggle 140k Real & Fake Faces (5,000 authentic FFHQ photographs vs 5,000 StyleGAN2 synthetic portraits).
- **Deduplication:** 64-bit 2D DCT Perceptual Hashing (pHash) ensuring zero duplicate leaks across splits.
- **Split Partitioning:** Stratified 70% Train (7,000 samples: 3,500 Real / 3,500 Fake), 15% Validation (1,500 samples), 15% Locked Test (1,500 samples).
- **Physics-Based Forensic Augmentations:**
  - Dynamic social media JPEG recompression ($Q \in [40, 92]$, $p=0.40$).
  - Physical camera sensor noise injection (Gaussian ISO grain $\sigma \in [0.003, 0.012]$, $p=0.30$).
  - Multi-scale face crop scale jitter ($[0.75, 0.95]$, $p=0.45$).
  - Random rotation ($\pm 10^\circ$), horizontal flip ($p=0.50$), and brightness/contrast jitter ($\pm 12\%$).
- **Training Strategy:** Two-stage transfer learning on CUDA (NVIDIA GeForce RTX 3050 Laptop GPU):
  - **Stage 1 (Head Warmup):** 3 epochs, backbone frozen, $\eta_{\text{head}} = 5 \times 10^{-4}$, AdamW.
  - **Stage 2 (Selective Fine-Tuning):** 18 epochs, selective unfreezing of stages 6 & 7 with differential learning rates ($\eta_{\text{backbone}} = 4 \times 10^{-5}, \eta_{\text{head}} = 3 \times 10^{-4}$), Cosine Annealing scheduler, balanced positive weight $w_{\text{pos}} = 0.95$, and label smoothing $\epsilon = 0.03$.
  - **Peak Checkpoint:** Best validation Macro-F1 reached **0.9580** (Val Accuracy: **95.8%**, Final Train Accuracy: **94.6%**). Checkpoint saved to `models/best_model.pth`.
- **Post-Hoc Probability Calibration:** Platt Temperature Scaling ($T = 0.9041$) fitted on validation logits via negative log-likelihood minimization using L-BFGS, reducing Expected Calibration Error (ECE) to **0.0118** and Brier Score to **0.0320**.

---

## Evaluation & Performance Summary
Detailed metrics evaluated on the locked held-out test partition (`reports/metrics.json`, 1,500 balanced samples):

| Metric | Measured Value | Target Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | **96.07%** | $\ge 90.0\%$ | Exceeded |
| **ROC-AUC** | **0.9922** | $\ge 0.920$ | Exceeded |
| **PR-AUC** | **0.9929** | $\ge 0.900$ | Exceeded |
| **Fake Recall / Sensitivity** | **96.93%** (727 / 750) | $\ge 90.0\%$ | Exceeded |
| **Real Specificity** | **95.20%** (714 / 750) | $\ge 90.0\%$ | Exceeded |
| **False Alarm Rate (FPR)** | **4.80%** (36 / 750) | $\le 10.0\%$ | Exceeded |
| **Macro F1-Score** | **0.9607** | $\ge 0.900$ | Exceeded |
| **Expected Calibration Error (ECE)** | **0.0118** | $\le 0.050$ | Exceeded |
| **Equal Error Rate (EER)** | **0.0387** (3.87%) | $\le 0.080$ | Exceeded |

### Operating Decision Modes:
* **⚡ Direct Verdict Mode (Default):** Single calibrated threshold $\tau = 0.58$. Scores $< 0.58$ verify as Real; scores $\ge 0.58$ flag as Synthetic. Delivers $96.07\%$ accuracy on clean and slightly blurred real portraits.
* **🔬 Forensic 3-Tier Mode (Abstain Band):** Strict forensic bounds ($s \le 0.35$ Verified Real, $s \ge 0.65$ Flagged Synthetic, $0.35 < s < 0.65$ Inconclusive/Abstain for manual analyst triage).

---

## Known Limitations & Failure Modes
1. **The Generalization Gap:** Generative AI techniques evolve rapidly; diffusion models with novel latent upsamplers may require periodic retraining.
2. **Extreme Facial Motion Blur:** Severe motion blur ($\sigma^2_{\text{Laplacian}} < 65$) can smooth out biological micro-pores; the system triggers a visible blur advisory banner.
3. **High-Frequency Woven Lattice Textures:** Certain real-world textures (straw hats, dense knitwear, moiré screen grids) generate periodic frequency patterns resembling GAN/diffusion checkerboard artifacts.
4. **Cosmetic Airbrushing & Commercial Studio Retouching:** Heavy digital skin smoothing and studio filters may trigger elevated synthetic scores.

---

## Demographic Considerations & Ethical Safeguards
- Underlying training datasets include real FFHQ images and StyleGAN2 generations.
- The system must not be deployed as an autonomous adjudicator to prevent disparate impact across demographic cohorts.
- High-stakes decisions mandate human-in-the-loop review supported by the downloadable Executive PDF Forensic Audit Certificate.
