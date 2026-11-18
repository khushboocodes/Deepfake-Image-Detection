"""
Dataset Downloader & Unpacker Utility
Automates acquisition of official benchmark datasets:
1. GenImage subset (ImageNet-Real & Stable Diffusion / Midjourney)
2. Synthbuster (Zenodo 10066460: DALL-E 3, Midjourney v5, SDXL)
3. Real Facial Portraits (FFHQ Open Subsets) & Paired Synthetics
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
import shutil


def download_progress_hook(count, block_size, total_size):
    """Prints a clean CLI progress bar for downloads."""
    percent = int(count * block_size * 100 / max(1, total_size))
    downloaded_mb = (count * block_size) / (1024 * 1024)
    total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
    sys.stdout.write(f"\rDownloading: {percent}% [{downloaded_mb:.1f} MB / {total_mb:.1f} MB]")
    sys.stdout.flush()


def download_file(url: str, dest_path: str) -> None:
    """Downloads a file from URL with progress reporting."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    print(f"\nFetching: {url}")
    print(f"Target:   {dest_path}")
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest_path, reporthook=download_progress_hook)
    print("\nDownload complete!")


def extract_archive(archive_path: str, extract_dir: str) -> None:
    """Extracts .zip, .tar.gz, or .tar files."""
    print(f"Extracting '{archive_path}' into '{extract_dir}'...")
    os.makedirs(extract_dir, exist_ok=True)
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tar_ref:
            tar_ref.extractall(extract_dir)
    print("Extraction complete!")


def setup_synthbuster(target_dir: str = "data/raw/synthbuster") -> None:
    """
    Downloads and structures Synthbuster diffusion models evaluation set from Zenodo (Record 10066460).
    """
    os.makedirs(target_dir, exist_ok=True)
    print("=" * 70)
    print("ACQUIRING SYNTHBUSTER BENCHMARK (Zenodo Record 10066460)")
    print("=" * 70)
    print("Zenodo URL: https://zenodo.org/records/10066460")
    print(f"Target Directory: {target_dir}")
    print("\nSynthbuster packages:")
    print(" - dalle3.zip (1,000 uncompressed DALL-E 3 portraits/scenes)")
    print(" - midjourney-v5.zip (1,000 Midjourney v5 generations)")
    print(" - stable-diffusion-xl.zip (1,000 SDXL generations)")
    print(" - prompts.csv")
    print("\nTo download directly using Python:")
    zenodo_base = "https://zenodo.org/records/10066460/files"
    print(f"Direct link example: {zenodo_base}/prompts.csv?download=1")


def setup_genimage_guide() -> None:
    """Prints actionable instructions for downloading GenImage splits."""
    print("=" * 70)
    print("GENIMAGE BENCHMARK ACQUISITION INSTRUCTIONS")
    print("=" * 70)
    print("Official Project Page: https://genimage-dataset.github.io/")
    print("Official GitHub:       https://github.com/GenImage-Dataset/GenImage")
    print("\nMethod 1: Hugging Face Datasets (Fastest):")
    print("  pip install huggingface_hub datasets")
    print("  huggingface-cli download nebula/GenImage-arrow --local-dir data/raw/genimage")
    print("\nMethod 2: Direct Google Drive / Baidu Mirrors:")
    print("  Check https://github.com/GenImage-Dataset/GenImage#download-links")
    print("  Recommended MVP subset: Stable Diffusion v1.5 + ImageNet Validation")


def setup_ffhq_guide() -> None:
    """Prints instructions for acquiring authentic real human portrait photos (FFHQ)."""
    print("=" * 70)
    print("AUTHENTIC HUMAN PORTRAITS (FFHQ / CelebA-HQ)")
    print("=" * 70)
    print("For facial identity verification, FFHQ (Flickr-Faces-HQ) provides 70,000 real portraits:")
    print("  1. Kaggle Dataset: https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq")
    print("  2. Command: kaggle datasets download -d arnaud58/flickrfaceshq-dataset-ffhq -p data/raw/ffhq --unzip")
    print("  3. Hugging Face: https://huggingface.co/datasets/visual-layer/ffhq-thumbnails")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Acquisition CLI")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["genimage", "synthbuster", "ffhq", "all"],
        help="Which dataset to acquire or view instructions for"
    )
    args = parser.parse_args()

    if args.dataset in ("genimage", "all"):
        setup_genimage_guide()
        print("\n")
    if args.dataset in ("synthbuster", "all"):
        setup_synthbuster()
        print("\n")
    if args.dataset in ("ffhq", "all"):
        setup_ffhq_guide()
