# Dataset Architecture, Curation & Governance Specification

## AI-Based Deepfake Image Detection System for Digital Identity Verification
**Document Version:** 1.0.0 (Comprehensive Forensic Dataset Specification)  
**Associated PRD:** [`docs/prd.md`](./prd.md)  
**Associated Implementation Tasks:** [`docs/tasks.md`](./tasks.md)  
**Operational Working Directory:** `c:\Users\loq\Downloads\Deepfake Image Detection`

---

## 1. Executive Overview & Dataset Taxonomy

### 1.1 The Role of Benchmark Datasets in Image Forensics
In machine learning-based digital forensics, dataset selection dictates the boundary between genuine forensic artifact detection and superficial shortcut learning. Modern synthetic media generation has evolved rapidly from early Generative Adversarial Networks (GANs) to sophisticated Latent Diffusion Models (LDMs), Transformer-based autoregressive generators, and fine-grained facial identity-swapping pipelines.

Detecting synthetic imagery in digital identity verification (KYC/AML, account onboarding, credential validation) requires training models on datasets that:
1. **Reflect Modern Generator Footprints:** Cover modern diffusion generators (Stable Diffusion, Midjourney, DALL-E) rather than obsolete GAN architectures exclusively.
2. **Prevent Shortcut Learning:** Ensure authentic (real) images and synthetic (fake) images do not differ systematically in resolution, color profiles, JPEG compression rates, or background watermarks.
3. **Guarantee Split Integrity:** Enforce strict group-aware and generator-aware partitioning to eliminate data leakage between training, validation, and testing splits.
4. **Support Generalization Benchmarking:** Include held-out, completely unseen generator families to evaluate out-of-distribution (OOD) cross-generator robustness.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              DATASET TAXONOMY & ROLE                                   │
├──────────────────────────────┬─────────────────────────────┬───────────────────────────┤
│ Dataset Name                 │ Primary Role in Project     │ Focus Manipulation Target │
├──────────────────────────────┼─────────────────────────────┼───────────────────────────┤
│ GenImage                     │ Primary Training & Internal │ Full-Image Latent         │
│ (Primary Benchmark)          │ Validation / Locked Test    │ Diffusion & GAN Synthesis │
├──────────────────────────────┼─────────────────────────────┼───────────────────────────┤
│ Synthbuster                  │ Held-Out External           │ Modern Diffusion Models   │
│ (Generalization Benchmark)   │ Generalization Evaluation   │ (DALL-E 3, Midjourney v5) │
├──────────────────────────────┼─────────────────────────────┼───────────────────────────┤
│ FaceForensics++              │ Contingency Biometric       │ Video Face-Swap /         │
│ (Facial Swap Fallback)       │ Manipulation Benchmark      │ Facial Reenactment        │
├──────────────────────────────┼─────────────────────────────┼───────────────────────────┤
│ Celeb-DF v2                  │ Contingency Biometric       │ High-Quality Face Swap    │
│ (Advanced Deepfake Fallback) │ Manipulation Benchmark      │ Video Frames              │
├──────────────────────────────┼─────────────────────────────┼───────────────────────────┤
│ IMD2020                      │ Deferred Forensic Analysis  │ Localized Splicing /      │
│ (Manipulation Benchmark)     │                             │ Inpainting / Copy-Move    │
├──────────────────────────────┼─────────────────────────────┼───────────────────────────┤
│ CIFAKE                       │ Evaluated & Rejected        │ Low-Resolution (32x32)    │
│ (Low-Res Benchmark)          │ (Unsuitable for Identity)   │ Synthetic CIFAR-10 Images │
└──────────────────────────────┴─────────────────────────────┴───────────────────────────┘
```

---

## 2. Primary Benchmark Dataset: GenImage

### 2.1 Overview & Official Metadata
- **Official Repository:** [https://genimage-dataset.github.io/](https://genimage-dataset.github.io/)
- **Source Code / GitHub:** [https://github.com/GenImage-Dataset/GenImage](https://github.com/GenImage-Dataset/GenImage)
- **Scientific Publication:** *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images* (Zhu et al., NeurIPS 2023 / arXiv:2306.08571).
- **Core Design:** GenImage pairs real images from the ImageNet-1K (ILSVRC2012) validation and training pools with synthetic images generated using text prompts matching ImageNet category labels across 8 distinct generative models.

### 2.2 Generator Families & Architecture Composition
GenImage spans 8 generator families encompassing diverse generative algorithms, latent resolutions, and neural decoders:

| Sub-Dataset / Generator | Underlying Architecture | Generative Family | Native Resolution | Prompt / Conditioning |
| :--- | :--- | :--- | :--- | :--- |
| **Stable Diffusion v1.4** | Latent Diffusion Model (LDM) | Diffusion | $512 \times 512$ | ImageNet class labels |
| **Stable Diffusion v1.5** | Latent Diffusion Model (LDM) | Diffusion | $512 \times 512$ | ImageNet class labels |
| **Midjourney (v4/v5)** | Proprietary Diffusion Pipeline | Commercial Diffusion | Variable ($>512$) | Curated high-aesthetic prompts |
| **ADM** (Ablated Diffusion) | Pixel-Space Guided Diffusion | Diffusion | $256 \times 256$ | Class-conditional |
| **GLIDE** | Cascaded Text-to-Image Diffusion | Diffusion | $256 \times 256$ | Text-guided classifier-free |
| **Wukong** | Chinese Large-Scale Diffusion | Diffusion | $512 \times 512$ | Multilingual text prompts |
| **VQDM** | Vector Quantized Diffusion | Discrete Diffusion | $256 \times 256$ | Latent discrete codes |
| **BigGAN** | Deep Large-Scale GAN | GAN | $256 \times 256$ / $512$ | Class-conditional vectors |
| **Real Baseline (ImageNet)**| Optical Camera Photographs | Physical Reality | Variable (Resized) | Natural photographic captures |

### 2.3 Curated MVP Dataset Subset Specification
Downloading the full 1.3 million images of GenImage (~120 GB) is impractical for a 3-day MVP delivery. Therefore, this project extracts and locks a representative, balanced subset of **10,000 images**:

```
GenImage Curated MVP Subset: 10,000 Total Images
 ├── Real Class (Label 0): 5,000 Images
 │    └── Source: ImageNet-1K (ILSVRC2012) Validation & Curated Photography
 │
 └── Synthetic Class (Label 1): 5,000 Images
      ├── Stable Diffusion v1.4 / v1.5: 2,500 Images (Diffusion baseline)
      ├── Midjourney: 1,500 Images (High-aesthetic commercial diffusion)
      └── BigGAN: 1,000 Images (Generative Adversarial Network representative)
```

#### Sampling Rationale
- **Class Balance:** Exactly $50\%$ Real ($5,000$) and $50\%$ Synthetic ($5,000$) to guarantee zero class-prevalence bias.
- **Generator Diversity:** Combining Diffusion models ($80\%$) and GANs ($20\%$) ensures the convolutional backbone does not overfit to a single decoder architecture.
- **Partitioning:**
  - **Training Split (70%):** $7,000$ images ($3,500$ real, $3,500$ synthetic).
  - **Validation Split (15%):** $1,500$ images ($750$ real, $750$ synthetic).
  - **Locked Test Split (15%):** $1,500$ images ($750$ real, $750$ synthetic).

---

## 3. External Generalization Benchmark: Synthbuster

### 3.1 Overview & Official Access
- **Official Repository / DOI:** Zenodo Record [10.5281/zenodo.10066460](https://zenodo.org/records/10066460)
- **Scientific Publication:** *Synthbuster: Towards Detection of Diffusion Model Generated Images* (Bammey et al., IEEE Access 2024 / arXiv:2311.16828).
- **License:** Creative Commons Attribution 4.0 International (CC-BY 4.0).
- **Core Mission:** A dedicated evaluation dataset designed explicitly to test whether deepfake detectors generalize to **unseen** state-of-the-art diffusion models.

### 3.2 Generator Composition & Out-of-Distribution Setup
Synthbuster contains 1,000 pristine camera-captured real images and pairs them with generations from 9 modern diffusion models. The project utilizes a held-out slice of **1,000 images** from the most advanced generators:

| Generator Model | Release Date | Architecture | Evaluation Role |
| :--- | :--- | :--- | :--- |
| **DALL-E 3** | Late 2023 | OpenAI Transformer-Diffusion | Unseen Zero-Shot Test |
| **DALL-E 2** | 2022 | CLIP-conditioned Diffusion | Unseen Zero-Shot Test |
| **Midjourney v5** | 2023 | Proprietary Diffusion | Unseen Zero-Shot Test |
| **Adobe Firefly** | 2023 | Commercial Stock Diffusion | Unseen Zero-Shot Test |
| **Stable Diffusion XL (SDXL)**| Mid 2023 | Large Latent Diffusion ($1024^2$ native) | Unseen Zero-Shot Test |
| **Pristine Real Images** | Varied | Professional DSLR Optical Photography | Real Control Baseline |

### 3.3 Protocol for Generalization Benchmarking
1. **Zero-Gradient Enforcement:** Models trained in this project are **never** updated on Synthbuster images.
2. **Zero-Threshold Tuning:** Decision thresholds ($\tau_{\text{real}} = 0.35, \tau_{\text{fake}} = 0.65$) and temperature calibration ($T$) are locked using the GenImage validation split and applied directly to Synthbuster.
3. **Generalization Gap Reporting:** The system explicitly computes the drop in Accuracy, ROC-AUC, and F1-score when transitioning from the locked GenImage test set to Synthbuster, diagnosing real-world transferability.

---

## 4. Comparative Evaluation of Alternate & Contingency Datasets

### 4.1 Detailed Evaluation Matrix

| Dataset | Modality | Primary Focus | Image / Video Count | Access / Terms | Assessment for Identity MVP |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GenImage** | Still Images | Latent Diffusion & GAN synthesis | 1,331,167 images | Open research terms | **PRIMARY SELECTION:** Comprehensive, modern, paired, directly aligns with full-image synthesis. |
| **Synthbuster** | Still Images | Cross-generator diffusion models | 10,000 images | Zenodo CC-BY 4.0 | **SELECTED EXTERNAL TEST:** Premier benchmark for evaluating zero-shot generalization. |
| **FaceForensics++** | Video / Frames | Deepfakes, Face2Face, FaceSwap, NeuralTextures | 1,000 videos / 1.8M frames | Academic request form | **CONTINGENCY FALLBACK:** Mandatory if project task shifts specifically to facial manipulation/swapping. Requires strict video-level splitting. |
| **Celeb-DF v2** | Video / Frames | Advanced deepfake face swaps | 5,639 videos | Academic license request | **CONTINGENCY FALLBACK:** High-quality face-swap dataset with reduced boundary artifacts. |
| **IMD2020** | Still Images | Real-world digital splicing & inpainting | 35,000 images | Academic open download | **DEFERRED:** Focuses on localized post-processing manipulation rather than full AI generation. |
| **CIFAKE** | Still Images | SD v1.4 vs CIFAR-10 | 120,000 images | Kaggle / Open | **REJECTED:** $32 \times 32$ pixel resolution lacks facial biometric and fine-grained forensic realism required for identity verification. |
| **WildDeepfake** | Still Images / Web | In-the-wild facial deepfakes | 707 videos / 65K images | Academic open | **DEFERRED:** High label noise and inconsistent web compression introduce unwanted confounds. |

### 4.2 Contingency Activation Protocol (Face-Swap Scope Shift)
If during project execution the evaluation panel requires focusing strictly on **facial biometric manipulation** (e.g., identity impersonation or face swapping) rather than full-image synthetic generation:
1. Deactivate GenImage and activate **FaceForensics++ (Deepfakes / FaceSwap subset)**.
2. Update task naming in all documentation to: `AI-Based Facial Manipulation Detection`.
3. **Mandatory Video-Level Splitting:** All frames extracted from source video `X` must reside in the exact same split (either Train, Validation, or Test). Splitting by individual frames is strictly prohibited due to severe near-duplicate identity leakage.

---

## 5. Dataset Governance, Legal Terms & Licensing

### 5.1 Legal Compliance & Licensing Matrix

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              DATASET LICENSING AUDIT                                   │
├──────────────────┬──────────────────────┬─────────────────┬────────────────────────────┤
│ Dataset          │ Upstream License     │ Commercial Use? │ Redistribution in Git?     │
├──────────────────┼──────────────────────┼─────────────────┼────────────────────────────┤
│ GenImage         │ Research / Academic  │ Restricted      │ STRICTLY PROHIBITED        │
│ (ImageNet/SD)    │ Permitted            │ (Non-commercial)│ (Store locally in data/)   │
├──────────────────┼──────────────────────┼─────────────────┼────────────────────────────┤
│ Synthbuster      │ Creative Commons     │ Attribution     │ PROHIBITED (Store locally  │
│                  │ CC-BY 4.0            │ Permitted       │ or download via Zenodo)    │
├──────────────────┼──────────────────────┼─────────────────┼────────────────────────────┤
│ FaceForensics++  │ Technical University │ Non-Commercial  │ STRICTLY PROHIBITED        │
│                  │ Munich Research Only │ Research Only   │ (Requires signed access)   │
├──────────────────┼──────────────────────┼─────────────────┼────────────────────────────┤
│ Project Codebase │ MIT License          │ Open Source     │ Permitted (GitHub public)  │
│ & Manifests      │                      │ Permitted       │ (Contains metadata only)   │
└──────────────────┴──────────────────────┴─────────────────┴────────────────────────────┘
```

### 5.2 Ethical AI, GDPR & Biometric Privacy Safeguards
1. **Zero Raw Image Uploads to Version Control:** The project `.gitignore` explicitly prevents any images under `data/raw/` or `data/processed/` from being tracked by Git.
2. **Ephemeral Identity Processing:** The application processes images in memory. No user-uploaded portrait is persisted to disk unless the user explicitly requests an export.
3. **Demographic Representation Audit:** The team acknowledges that ImageNet-derived and early Stable Diffusion datasets carry known demographic imbalances (skewing toward Western skin tones and celebrities). The model must be documented as an academic diagnostic prototype and prohibited from autonomous identity denial under GDPR Article 22.

---

## 6. Data Engineering, Ingestion & Deduplication Pipeline

### 6.1 Cryptographic Verification & Ingestion Protocol
1. **Raw Storage Immutability:** Downloaded archives are unpacked into `data/raw/<dataset_name>/` and assigned read-only permissions.
2. **Integrity Verification:** Every ingested image is hashed via SHA-256 upon ingestion to establish cryptographic provenance:
   $$\text{SHA-256}(B) = \text{HexDigest of binary payload}$$
3. **Format Validation:** Non-decodable files, zero-byte artifacts, and unsupported formats are purged during the initial ingestion pass.

### 6.2 Perceptual Hash Deduplication Protocol
Near-duplicate images (e.g., identical portraits with slight color adjustments or crop variations) can cause catastrophic train-to-test leakage if split across sets.

```
Input Image 
     │
     ├─ Step 1: Downsample to 8x8 Grayscale
     ├─ Step 2: Compute Discrete Cosine Transform (DCT)
     ├─ Step 3: Extract 8x8 Low-Frequency Matrix (excluding DC term)
     ├─ Step 4: Compute Median Frequency
     ├─ Step 5: Generate 64-bit Binary Hash (Bit=1 if DCT > Median, else 0)
     └─ Step 6: Compute Hamming Distance H(h1, h2) across all pairs:
                H(h1, h2) = sum(b1 XOR b2)
```

- **Decontamination Rule:** Any pair with Hamming distance $H(h_1, h_2) \le 3$ is flagged. If detected, both images are either assigned to the **exact same split** (e.g., both in Train) or the duplicate copy is pruned from the dataset.

### 6.3 Generator-Aware Stratified Partitioning Protocol
Splitting must maintain balanced class representations and balanced generator distributions across Train (70%), Validation (15%), and Test (15%).

```python
# Partitioning Stratification Matrix
# Each stratum S(c, g) is divided into 70/15/15:
For each category c in {Real, Synthetic}:
    For each generator g in {ImageNet, SD_v1.4, SD_v1.5, Midjourney, BigGAN}:
        Subset = Filter(dataset, label==c, generator==g)
        Shuffle(Subset, seed=42)
        Train_Split.extend(Subset[0 : 0.70 * N])
        Val_Split.extend(Subset[0.70 * N : 0.85 * N])
        Test_Split.extend(Subset[0.85 * N : 1.00 * N])
```

#### Split Allocation Table

| Dataset Split | Real Count | Synthetic (SD) | Synthetic (MJ) | Synthetic (BigGAN) | Total Images | Allocation % | Purpose |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | 3,500 | 1,750 | 1,050 | 700 | **7,000** | 70.0% | Backpropagation & gradient updates |
| **Validation** | 750 | 375 | 225 | 150 | **1,500** | 15.0% | Checkpoint selection, LR schedule, calibration |
| **Locked Test**| 750 | 375 | 225 | 150 | **1,500** | 15.0% | Held-out final metric reporting (read once) |
| **External** | 500 | 0 (SDXL: 250)| 0 (MJv5: 250) | 0 (DALL-E: 250) | **1,000** | External | Zero-shot generalization evaluation |

---

## 7. Dataset Manifest Schema & Data Catalog

### 7.1 JSON Schema Definition (`manifests/dataset_manifest.json`)
All processed datasets are tracked via structured manifests generated by `src/dataset.py`.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "DatasetManifest",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "asset_id": { "type": "string", "description": "Unique identifier (UUID or SHA-256 prefix)" },
      "relative_path": { "type": "string", "description": "Relative file path from project root" },
      "source_dataset": { "type": "string", "enum": ["GenImage", "Synthbuster", "Custom_Demo"] },
      "generator_family": { "type": "string", "enum": ["ImageNet_Real", "StableDiffusion_v1.4", "StableDiffusion_v1.5", "Midjourney", "BigGAN", "DALL-E_3", "SDXL"] },
      "label": { "type": "integer", "enum": [0, 1], "description": "0 = Real/Authentic, 1 = Synthetic/AI-Generated" },
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
    "required": ["asset_id", "relative_path", "source_dataset", "generator_family", "label", "split", "sha256", "dimensions", "filesize_bytes"]
  }
}
```

### 7.2 Sample Manifest Entries

```json
[
  {
    "asset_id": "genimg_real_004812",
    "relative_path": "data/processed/train/real/ILSVRC2012_val_00004812.JPEG",
    "source_dataset": "GenImage",
    "generator_family": "ImageNet_Real",
    "label": 0,
    "split": "train",
    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "dimensions": [500, 375],
    "original_format": "JPEG",
    "filesize_bytes": 48210
  },
  {
    "asset_id": "genimg_sd15_001924",
    "relative_path": "data/processed/train/fake/sd15_portrait_001924.png",
    "source_dataset": "GenImage",
    "generator_family": "StableDiffusion_v1.5",
    "label": 1,
    "split": "train",
    "sha256": "8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4",
    "dimensions": [512, 512],
    "original_format": "PNG",
    "filesize_bytes": 384192
  },
  {
    "asset_id": "synth_dalle3_000412",
    "relative_path": "data/processed/external/dalle3_portrait_000412.png",
    "source_dataset": "Synthbuster",
    "generator_family": "DALL-E_3",
    "label": 1,
    "split": "external_test",
    "sha256": "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
    "dimensions": [1024, 1024],
    "original_format": "PNG",
    "filesize_bytes": 1048576
  }
]
```

---

## 8. Physical Directory Architecture on Disk

```
data/
├── raw/                                # Read-only original downloaded data (NOT committed)
│   ├── genimage/
│   │   ├── imagenet_val/               # Real photographs
│   │   ├── stable_diffusion_v1_4/      # SD v1.4 synthetic
│   │   ├── stable_diffusion_v1_5/      # SD v1.5 synthetic
│   │   ├── midjourney/                 # Midjourney synthetic
│   │   └── biggan/                     # BigGAN synthetic
│   └── synthbuster/
│       ├── pristine_real/              # 1,000 real photos
│       ├── dalle3/                     # DALL-E 3 synthetic
│       ├── sdxl/                       # SDXL synthetic
│       └── firefly/                    # Firefly synthetic
│
├── processed/                          # Curated & standardized subset
│   ├── train/
│   │   ├── real/                       # 3,500 real images
│   │   └── fake/                       # 3,500 synthetic images
│   ├── val/
│   │   ├── real/                       # 750 real images
│   │   └── fake/                       # 750 synthetic images
│   ├── test/
│   │   ├── real/                       # 750 real images (LOCKED)
│   │   └── fake/                       # 750 synthetic images (LOCKED)
│   └── external/
│       ├── real/                       # 500 pristine real images
│       └── fake/                       # 500 unseen diffusion images
│
└── manifests/                          # Versioned JSON manifest catalogs
    ├── dataset_manifest.json           # Master manifest of all assets
    ├── train_manifest.json             # 70% Training partition
    ├── val_manifest.json               # 15% Validation partition
    ├── test_manifest.json              # 15% Locked test partition
    └── external_manifest.json          # Synthbuster generalization partition
```

---

## 9. Forensic Quality Control & Known Dataset Pitfalls

### 9.1 Shortcut Learning Prevention Checklist
1. **Watermark Auditing:** Midjourney generations occasionally contain faint corner text or upscale artifacts. The curation script performs automated edge-crop auditing to ensure models do not trigger purely on corner watermark patterns.
2. **Compression Normalization:** Real ImageNet images have variable JPEG compression history, whereas raw diffusion samples are exported as uncompressed PNGs. The preprocessing pipeline standardizes all images to normalized float tensors ($224 \times 224$), and training augmentations apply subtle JPEG compression ($Q \in [70, 95]$) to eliminate PNG-vs-JPEG container shortcuts.
3. **Color Space Standardization:** Synthetic generators often omit ICC color profile chunks or default to untagged sRGB. The pipeline explicitly converts all images to standard 3-channel RGB before feature extraction.

### 9.2 Label Noise Mitigation
- In uncurated web deepfake datasets, images are frequently mislabeled due to automated scraping heuristics.
- By utilizing GenImage and Synthbuster—which generate synthetic images directly from controlled API calls and local model weights—ground truth labels are deterministic and carry zero label noise.

---

## 10. Verification & Quality Assurance Commands

To verify that the dataset manifests, folder layouts, and class balances adhere to this specification, execute:

```powershell
# 1. Verify existence of manifests
Test-Path manifests/train_manifest.json, manifests/val_manifest.json, manifests/test_manifest.json, manifests/external_manifest.json

# 2. Verify exact class balance in training split (Expected: 3500 real, 3500 fake)
python -c "import json; m = json.load(open('manifests/train_manifest.json')); reals = sum(1 for x in m if x['label']==0); fakes = sum(1 for x in m if x['label']==1); print(f'Train Balance: Real={reals}, Fake={fakes}'); assert reals == fakes"

# 3. Verify zero leakage between train and test splits
python -c "import json; tr = {x['sha256'] for x in json.load(open('manifests/train_manifest.json'))}; te = {x['sha256'] for x in json.load(open('manifests/test_manifest.json'))}; overlap = tr.intersection(te); assert len(overlap) == 0, f'LEAKAGE DETECTED: {len(overlap)} images'; print('Zero Split Leakage Verified: OK')"
```

*This document serves as the binding dataset governance and engineering reference. Any modifications to dataset sampling or split ratios must be recorded through manifest versioning.*
