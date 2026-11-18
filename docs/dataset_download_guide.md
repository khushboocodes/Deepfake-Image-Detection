# Step-by-Step Dataset Download & Ingestion Guide

## AI-Based Deepfake Image Detection System for Digital Identity Verification
**Document Version:** 1.0.0 (Official Benchmark Acquisition Manual)  
**Associated PRD:** [`docs/prd.md`](./prd.md)  
**Associated Datasets Spec:** [`docs/datasets.md`](./datasets.md)  
**Associated Runbook:** [`docs/developer_guide.md`](./developer_guide.md)  
**Operational Working Directory:** `c:\Users\loq\Downloads\Deepfake Image Detection`

---

## 1. Overview & Storage Planning

To train and evaluate on real-world benchmark datasets rather than the local starter set, download the official datasets into `data/raw/`. The automated manifest pipeline ([src/dataset.py](file:///c:/Users/loq/Downloads/Deepfake%20Image%20Detection/src/dataset.py)) indexes, deduplicates via perceptual hashing (`pHash`), and partitions the images into 70% Train, 15% Validation, and 15% Locked Test sets.

```
data/raw/
├── genimage/                 # Primary Benchmark (Real vs SD/Midjourney)
│   ├── imagenet_val/         # Real optical photographs
│   ├── stable_diffusion/     # Stable Diffusion synthetic images
│   └── midjourney/           # Midjourney synthetic images
├── synthbuster/              # External Generalization Benchmark
│   ├── dalle3/               # DALL-E 3 synthetic
│   ├── midjourney_v5/        # Midjourney v5 synthetic
│   └── pristine_real/        # Pristine camera photographs
└── ffhq/                     # Real Human Facial Portraits (70,000 real photos)
```

### Storage Requirements:
- **Recommended MVP Slice (10,000 images):** ~3 to 5 GB free disk space.
- **Large Benchmark (50,000 images):** ~15 to 25 GB free disk space.
- **Full 1.3M GenImage Archive:** ~120 GB (Only needed for cluster training).

---

## 2. Dataset 1: GenImage Benchmark (Primary Training Benchmark)

GenImage pairs real photographs from ImageNet-1K with synthetic images generated across 8 generative models (Stable Diffusion v1.4/v1.5, Midjourney, BigGAN, ADM, GLIDE, Wukong, VQDM).

- **Official Project Website:** [https://genimage-dataset.github.io/](https://genimage-dataset.github.io/)
- **Official GitHub:** [https://github.com/GenImage-Dataset/GenImage](https://github.com/GenImage-Dataset/GenImage)
- **Paper Citation:** *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images* (NeurIPS 2023)

### Method A: Fastest Download via Hugging Face (Recommended)
The community hosts pre-packaged subsets of GenImage on Hugging Face. You can download directly using the Hugging Face CLI:

```powershell
# 1. Install Hugging Face Hub tooling
pip install huggingface_hub

# 2. Download the Stable Diffusion v1.4 / v1.5 split and Real baseline
huggingface-cli download nebula/GenImage-arrow --repo-type dataset --include "data/train/SDV1.4/*" "data/val/SDV1.4/*" --local-dir data/raw/genimage

# 3. Alternatively, download the complete Arrow repository:
huggingface-cli download nebula/GenImage-arrow --repo-type dataset --local-dir data/raw/genimage
```

### Method B: Direct Python Stream Download
If you want to extract a clean 5,000-image balanced slice directly into `data/raw/genimage/`:

```python
from pathlib import Path
from huggingface_hub import snapshot_download

# Downloads the Stable Diffusion subset of GenImage
snapshot_dir = snapshot_download(
    repo_id="nebula/GenImage-arrow",
    repo_type="dataset",
    allow_patterns=["data/train/SDV1.4/*", "data/test/SDV1.4/*"],
    local_dir="data/raw/genimage"
)
print(f"Downloaded to: {snapshot_dir}")
```

### Method C: Official Google Drive & Baidu Netdisk Mirrors
If Hugging Face is blocked on your network, use the official university mirrors hosted by Shanghai Jiao Tong University:
1. Navigate to: [https://github.com/GenImage-Dataset/GenImage#download-links](https://github.com/GenImage-Dataset/GenImage#download-links)
2. Locate the Google Drive or Baidu folders for:
   - `imagenet_ai_0424_sdv5.tar.gz` (Stable Diffusion v1.5 subset)
   - `imagenet_ai_0419_midjourney.tar.gz` (Midjourney subset)
   - `ILSVRC2012_val.tar` (ImageNet Real baseline)
3. Download the archives and unpack them into `data/raw/genimage/`.

---

## 3. Dataset 2: Synthbuster Benchmark (External Generalization Test)

Synthbuster is hosted openly on Zenodo under a **CC-BY 4.0** license. It provides 1,000 uncompressed images each from modern diffusion models (DALL-E 3, Midjourney v5, SDXL, Adobe Firefly) paired with pristine optical photographs from RAISE-1k.

- **Zenodo Record:** [https://zenodo.org/records/10066460](https://zenodo.org/records/10066460)
- **DOI:** `10.5281/zenodo.10066460`
- **Paper Citation:** *Synthbuster: Towards Detection of Diffusion Model Generated Images* (IEEE Access 2024)

### Method A: Download via Browser
1. Open [https://zenodo.org/records/10066460](https://zenodo.org/records/10066460) in your web browser.
2. Scroll to the **Files** section.
3. Download:
   - `dalle3.zip` (1,000 images, ~1.4 GB)
   - `midjourney-v5.zip` (1,000 images, ~1.5 GB)
   - `stable-diffusion-xl.zip` (1,000 images, ~1.3 GB)
   - `prompts.csv`
4. Unpack the ZIP files into `data/raw/synthbuster/`.

### Method B: Automated PowerShell / Curl Download

```powershell
# Create target directory
New-Item -ItemType Directory -Force -Path data/raw/synthbuster | Out-Null

# Download DALL-E 3 slice
curl -L -o data/raw/synthbuster/dalle3.zip "https://zenodo.org/records/10066460/files/dalle3.zip?download=1"

# Download Midjourney v5 slice
curl -L -o data/raw/synthbuster/midjourney-v5.zip "https://zenodo.org/records/10066460/files/midjourney-v5.zip?download=1"

# Download SDXL slice
curl -L -o data/raw/synthbuster/stable-diffusion-xl.zip "https://zenodo.org/records/10066460/files/stable-diffusion-xl.zip?download=1"

# Unzip all downloaded files
Expand-Archive -Path data/raw/synthbuster/dalle3.zip -DestinationPath data/raw/synthbuster/dalle3
Expand-Archive -Path data/raw/synthbuster/midjourney-v5.zip -DestinationPath data/raw/synthbuster/midjourney_v5
Expand-Archive -Path data/raw/synthbuster/stable-diffusion-xl.zip -DestinationPath data/raw/synthbuster/sdxl
```

---

## 4. Dataset 3: Authentic Human Facial Portraits (FFHQ & StyleGAN Faces)

For **Digital Identity Verification (KYC)**, high-resolution portrait photographs of real humans are essential:

### 1. Real Human Faces: FFHQ (Flickr-Faces-HQ by NVIDIA)
NVIDIA's FFHQ contains 70,000 high-quality $1024 \times 1024$ real human portraits featuring varied ethnicities, ages, accessories (glasses, hats), and camera lighting.

- **Option 1: Kaggle CLI (Fastest):**
  ```powershell
  # Install kaggle tool
  pip install kaggle
  
  # Download 1024x1024 FFHQ dataset directly into data/raw/ffhq
  kaggle datasets download -d arnaud58/flickrfaceshq-dataset-ffhq -p data/raw/ffhq --unzip
  ```
- **Option 2: Hugging Face (128x128 Thumbnails - Lightweight):**
  ```powershell
  huggingface-cli download visual-layer/ffhq-thumbnails --repo-type dataset --local-dir data/raw/ffhq_thumbnails
  ```

### 2. Synthetic Human Faces: 100K-Generated-Faces / StyleGAN
Paired synthetic portraits generated by StyleGAN2/StyleGAN3:
- **Kaggle Dataset:** [https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)
- **Download Command:**
  ```powershell
  kaggle datasets download -d ciplab/real-and-fake-face-detection -p data/raw/faces --unzip
  ```

---

## 5. Dataset 4: Face-Swap & Video Manipulations (FaceForensics++ & Celeb-DF)

If your research focus shifts specifically to video face-swaps:

1. **FaceForensics++ (Technical University of Munich):**
   - Fill out the official academic license agreement form at: [https://github.com/ondyari/faceforensics](https://github.com/ondyari/faceforensics)
   - You will receive an automated download token. Run the official script:
     ```powershell
     python download-FaceForensics.py data/raw/faceforensics -d "Deepfakes" -c "c23" -t "videos"
     ```
2. **Celeb-DF v2 (State University of New York at Buffalo):**
   - Request access at: [https://cse.buffalo.edu/~siweilyu/celeb-deepfakeforensics.html](https://cse.buffalo.edu/~siweilyu/celeb-deepfakeforensics.html)

---

## 6. What to Do After Downloading: The Manifest Pipeline

Once you have placed your downloaded real and fake images into `data/raw/`:

### Step 1: Automatically Index and Partition the Data
Run the manifest builder. It inspects all images, computes SHA-256 hashes, runs perceptual deduplication (`pHash`), and creates balanced 70% Train, 15% Val, and 15% Locked Test splits:

```powershell
python src/dataset.py --action create_manifest --raw_dir data/raw --output_dir manifests/
```

### Step 2: Retrain the Neural Network Backbone
Run the two-stage transfer learning pipeline:

```powershell
python src/train.py --config configs/train_config.json --preprocess configs/preprocess.json
```
*Saves the new trained weights to `models/best_model.pth` and records epoch history to `reports/training_history.json`.*

### Step 3: Fit Temperature Scaling Probability Calibration
Calibrate output probabilities on the newly trained validation set:

```powershell
python src/calibration.py --val_manifest manifests/val_manifest.json --weights models/best_model.pth
```
*Updates `configs/preprocess.json` with the optimal temperature parameter $T$.*

### Step 4: Run Locked Test Set Evaluation
Evaluate the model against the locked held-out test split:

```powershell
python src/evaluate.py --test_manifest manifests/test_manifest.json --weights models/best_model.pth
```
*Generates updated confusion matrices, ROC-AUC plots, PR curves, and `reports/metrics.json`.*

### Step 5: Test Out-of-Distribution Generalization on Synthbuster
```powershell
python src/evaluate.py --test_manifest manifests/external_manifest.json --weights models/best_model.pth --output_prefix generalization_
```

### Step 6: Launch Web Interface
```powershell
streamlit run app.py
```

*All scripts are fully automated and designed to process whatever dataset you place inside `data/raw/`.*
