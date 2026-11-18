"""
Dataset Engineering, Manifest Management, and PyTorch DataLoader Module
Handles cryptographic asset indexing, perceptual deduplication, generator-aware
stratified partitioning, and forensic PyTorch Dataset loading.
"""

import os
import sys
import io
import json
import random
import hashlib

# Ensure project root is in sys.path for direct CLI execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.fftpack import dct
import PIL.Image
import PIL.ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from src.preprocess import (
    normalize_exif_orientation,
    ensure_three_channel_rgb,
    resize_image_deterministic,
    image_to_normalized_tensor
)


def compute_phash_numpy(image: PIL.Image.Image) -> str:
    """
    Computes a 64-bit perceptual hash (pHash) using discrete cosine transform (DCT) logic.
    Works natively via NumPy and SciPy without requiring external imagehash dependencies.
    """
    img_gray = image.convert("L").resize((32, 32), resample=PIL.Image.Resampling.BILINEAR)
    arr = np.array(img_gray, dtype=np.float32)
    
    # 2D DCT using separable 1D DCT formulation
    dct_rows = dct(arr, axis=0, norm="ortho")
    dct_2d = dct(dct_rows, axis=1, norm="ortho")
    
    # Extract 8x8 top-left low-frequency quadrant, excluding DC component (0,0)
    low_freq = dct_2d[:8, :8]
    median_val = np.median(low_freq[1:, 1:])
    
    # Build 64-bit binary string
    bits = low_freq > median_val
    hash_hex = f"{int(''.join(bits.flatten().astype(int).astype(str)), 2):016x}"
    return hash_hex


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 cryptographic digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_splits(
    dataset_manifest: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Partitions a dataset manifest into Train, Validation, and Locked Test splits
    using generator-aware and class-stratified sampling.
    """
    rng = random.Random(seed)
    
    # Group items by stratum key: (label, generator_family)
    strata: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for item in dataset_manifest:
        key = (item["label"], item.get("generator_family", "unknown"))
        strata.setdefault(key, []).append(item)
        
    train_split: List[Dict[str, Any]] = []
    val_split: List[Dict[str, Any]] = []
    test_split: List[Dict[str, Any]] = []
    
    for key, items in strata.items():
        items_copy = list(items)
        rng.shuffle(items_copy)
        
        n_total = len(items_copy)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        
        # Ensure at least 1 item in val and test if stratum has enough samples
        if n_total >= 3:
            n_train = max(1, min(n_train, n_total - 2))
            n_val = max(1, min(n_val, n_total - n_train - 1))
            
        train_items = items_copy[:n_train]
        val_items = items_copy[n_train:n_train + n_val]
        test_items = items_copy[n_train + n_val:]
        
        for x in train_items:
            x["split"] = "train"
        for x in val_items:
            x["split"] = "val"
        for x in test_items:
            x["split"] = "test"
            
        train_split.extend(train_items)
        val_split.extend(val_items)
        test_split.extend(test_items)
        
    rng.shuffle(train_split)
    rng.shuffle(val_split)
    rng.shuffle(test_split)
    
    return train_split, val_split, test_split


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset implementing forensic-preserving transforms for training and evaluation.
    """
    def __init__(
        self,
        manifest: List[Dict[str, Any]],
        base_dir: str = "",
        is_training: bool = False,
        img_size: Tuple[int, int] = (224, 224)
    ):
        self.manifest = manifest
        self.base_dir = base_dir
        self.is_training = is_training
        self.img_size = img_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.manifest[idx]
        rel_path = item["relative_path"]
        filepath = os.path.join(self.base_dir, rel_path) if self.base_dir else rel_path
        asset_id = item.get("asset_id", os.path.basename(rel_path))
        label_val = float(item["label"])

        try:
            with open(filepath, "rb") as f:
                img_bytes = f.read()
            pil_img = PIL.Image.open(io.BytesIO(img_bytes))
            pil_img.load()
        except Exception:
            # Fallback zero-filled image if file read fails
            pil_img = PIL.Image.new("RGB", self.img_size, (128, 128, 128))

        # Forensic pipeline
        pil_img = normalize_exif_orientation(pil_img)
        pil_img = ensure_three_channel_rgb(pil_img)

        if self.is_training:
            # SOTA Physical Constraint Simulation Suite (Option A):
            # 1. Multi-scale face-centric crop (p=0.50): exposes model to fine-grained skin micro-textures and wider context
            if random.random() < 0.50:
                w, h = pil_img.size
                scale = random.uniform(0.65, 0.98)
                new_w, new_h = max(32, int(w * scale)), max(32, int(h * scale))
                if new_w < w and new_h < h:
                    left = random.randint(0, w - new_w)
                    top = random.randint(0, h - new_h)
                    pil_img = pil_img.crop((left, top, left + new_w, top + new_h))

            # 2. Defocus & Lens Softness Simulation (p=0.35):
            # Inoculates model against mistaking motion blur or soft optical focus for AI smoothing
            if random.random() < 0.35:
                blur_radius = random.uniform(0.4, 1.8)
                pil_img = pil_img.filter(PIL.ImageFilter.GaussianBlur(radius=blur_radius))

            # 3. Overhead Sunhat / Ambient Lighting Shadow Gradient (p=0.25):
            # Simulates hats, overhead sunlight, or uneven directional illumination across the face
            if random.random() < 0.25:
                try:
                    w, h = pil_img.size
                    arr = np.array(pil_img, dtype=np.float32)
                    top_factor = random.uniform(0.65, 0.88)
                    grad = np.linspace(top_factor, 1.0, h, dtype=np.float32)[:, None, None]
                    arr = np.clip(arr * grad, 0, 255).astype(np.uint8)
                    pil_img = PIL.Image.fromarray(arr)
                except Exception:
                    pass

            # 4. Random horizontal flip (p=0.50)
            if random.random() < 0.50:
                pil_img = TF.hflip(pil_img)

            # 5. Mild rotation (-10 to +10 degrees, p=0.35)
            if random.random() < 0.35:
                angle = random.uniform(-10.0, 10.0)
                pil_img = TF.rotate(pil_img, angle, interpolation=TF.InterpolationMode.BILINEAR)

            # 6. Dynamic Social Media / WhatsApp JPEG Compression Simulation (p=0.45)
            if random.random() < 0.45:
                try:
                    compress_buf = io.BytesIO()
                    q = random.randint(35, 92)
                    pil_img.save(compress_buf, format="JPEG", quality=q)
                    compress_buf.seek(0)
                    pil_img = PIL.Image.open(compress_buf)
                    pil_img.load()
                except Exception:
                    pass

            # 7. Subtle Color, Contrast & Gamma variations (p=0.30)
            if random.random() < 0.30:
                bright_factor = random.uniform(0.90, 1.10)
                pil_img = TF.adjust_brightness(pil_img, bright_factor)
                contrast_factor = random.uniform(0.90, 1.10)
                pil_img = TF.adjust_contrast(pil_img, contrast_factor)

        # Standardized bilinear resizing with antialiasing
        pil_img = resize_image_deterministic(pil_img, target_size=self.img_size, interpolation="bilinear")

        # Convert to tensor
        tensor = TF.to_tensor(pil_img)

        # 8. Physical Camera Sensor Noise Augmentation (Gaussian ISO grain simulation on tensor, p=0.35)
        if self.is_training and random.random() < 0.35:
            noise_std = random.uniform(0.003, 0.015)
            noise = torch.randn_like(tensor) * noise_std
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)

        tensor = TF.normalize(tensor, mean=self.mean, std=self.std)

        label_tensor = torch.tensor([label_val], dtype=torch.float32)
        return tensor, label_tensor, asset_id


def create_dataloader(
    manifest: List[Dict[str, Any]],
    batch_size: int = 32,
    is_training: bool = False,
    num_workers: int = 0,
    base_dir: str = ""
) -> DataLoader:
    """Factory helper to construct a configured PyTorch DataLoader."""
    dataset = DeepfakeDataset(
        manifest=manifest,
        base_dir=base_dir,
        is_training=is_training
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )


def create_manifest_from_raw_dir(
    raw_dir: str = "data/raw",
    output_dir: str = "manifests",
    seed: int = 42,
    max_samples_per_class: Optional[int] = None
) -> Dict[str, Any]:
    """
    Scans raw directory for real and fake images, performs perceptual deduplication,
    generates stratified splits, and saves manifest JSON files.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    raw_path = os.path.abspath(raw_dir)
    print(f"Scanning for images in: {raw_path}")

    raw_items = []
    seen_hashes = set()
    duplicates_skipped = 0
    count_real = 0
    count_fake = 0

    # Recursive scan
    for root, dirs, files in os.walk(raw_path):
        # Ignore hidden folders
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        
        for file in files:
            if max_samples_per_class is not None:
                if count_real >= max_samples_per_class and count_fake >= max_samples_per_class:
                    break

            ext = os.path.splitext(file)[1].lower()
            if ext not in valid_extensions:
                continue

            filepath = os.path.join(root, file)
            # Use relative path from raw_dir so that parent project folders (like 'Deepfake Image Detection') don't pollute the label check
            rel_from_raw = os.path.relpath(filepath, raw_path).lower().replace("\\", "/")

            # Label assignment based on directory path inside raw_dir
            if "fake" in rel_from_raw or "synthetic" in rel_from_raw:
                label = 1
                gen_family = "Synthetic_Face"
                if max_samples_per_class is not None and count_fake >= max_samples_per_class:
                    continue
            elif "real" in rel_from_raw or "authentic" in rel_from_raw:
                label = 0
                gen_family = "Real_Face"
                if max_samples_per_class is not None and count_real >= max_samples_per_class:
                    continue
            else:
                continue

            try:
                with PIL.Image.open(filepath) as img:
                    img.load()
                    width, height = img.size
                    orig_format = img.format or "JPEG"
                    phash_val = compute_phash_numpy(img)
            except Exception as e:
                continue

            # Perceptual deduplication check
            hash_key = (label, phash_val)
            if hash_key in seen_hashes:
                duplicates_skipped += 1
                continue
            seen_hashes.add(hash_key)

            if label == 0:
                count_real += 1
            else:
                count_fake += 1

            file_sha = compute_sha256(filepath)
            rel_path = os.path.relpath(filepath, ".").replace("\\", "/")
            raw_items.append({
                "asset_id": f"{gen_family}_{len(raw_items):05d}",
                "relative_path": rel_path,
                "source_dataset": "Face_Dataset",
                "generator_family": gen_family,
                "label": label,
                "sha256": file_sha,
                "phash": phash_val,
                "dimensions": [width, height],
                "original_format": orig_format,
                "filesize_bytes": os.path.getsize(filepath)
            })

    print(f"Discovered {len(raw_items)} unique images (Skipped {duplicates_skipped} duplicates).")
    if not raw_items:
        print(f"No valid labeled images found in '{raw_dir}'. Make sure images are in folders with 'real' or 'fake' in their names.")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    train_split, val_split, test_split = generate_splits(raw_items, seed=seed)

    dataset_manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    train_manifest_path = os.path.join(output_dir, "train_manifest.json")
    val_manifest_path = os.path.join(output_dir, "val_manifest.json")
    test_manifest_path = os.path.join(output_dir, "test_manifest.json")

    all_manifest = train_split + val_split + test_split
    with open(dataset_manifest_path, "w") as f:
        json.dump(all_manifest, f, indent=2)
    with open(train_manifest_path, "w") as f:
        json.dump(train_split, f, indent=2)
    with open(val_manifest_path, "w") as f:
        json.dump(val_split, f, indent=2)
    with open(test_manifest_path, "w") as f:
        json.dump(test_split, f, indent=2)

    real_count = sum(1 for x in all_manifest if x["label"] == 0)
    fake_count = sum(1 for x in all_manifest if x["label"] == 1)

    print(f"Manifests successfully generated in '{output_dir}/':")
    print(f" - Total: {len(all_manifest)} (Real: {real_count}, Fake: {fake_count})")
    print(f" - Train: {len(train_split)}")
    print(f" - Val:   {len(val_split)}")
    print(f" - Test:  {len(test_split)}")

    return {
        "total": len(all_manifest),
        "train": len(train_split),
        "val": len(val_split),
        "test": len(test_split)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset engineering and manifest generator CLI.")
    parser.add_argument("--action", type=str, default="create_manifest", choices=["create_manifest", "verify_splits"])
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw image directory")
    parser.add_argument("--output_dir", type=str, default="manifests", help="Output directory for manifests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for partitioning")
    parser.add_argument("--max_samples_per_class", type=int, default=None, help="Optional maximum unique samples per class")
    args = parser.parse_args()

    if args.action == "create_manifest":
        create_manifest_from_raw_dir(
            args.raw_dir,
            args.output_dir,
            seed=args.seed,
            max_samples_per_class=args.max_samples_per_class
        )
    else:
        print(f"Dataset CLI initialized: Action={args.action}")
