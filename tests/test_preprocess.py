"""
Unit and Integration Tests for Forensic Preprocessing and Metadata Harvesting Module.
"""

import pytest
import numpy as np
import PIL.Image
import torch
from src.preprocess import (
    extract_forensic_metadata,
    normalize_exif_orientation,
    ensure_three_channel_rgb,
    resize_image_deterministic,
    image_to_normalized_tensor,
    compute_image_quality_metrics
)


def test_extract_forensic_metadata():
    img = PIL.Image.new("RGB", (100, 100))
    meta = extract_forensic_metadata(img)
    assert isinstance(meta, dict)
    assert "has_exif" in meta
    assert "icc_profile_present" in meta
    assert meta["has_exif"] is False


def test_ensure_three_channel_rgb():
    # RGB passthrough
    rgb = PIL.Image.new("RGB", (50, 50), (255, 0, 0))
    res_rgb = ensure_three_channel_rgb(rgb)
    assert res_rgb.mode == "RGB"

    # Grayscale conversion
    gray = PIL.Image.new("L", (50, 50), 128)
    res_gray = ensure_three_channel_rgb(gray)
    assert res_gray.mode == "RGB"

    # RGBA transparency composite over white
    rgba = PIL.Image.new("RGBA", (50, 50), (0, 0, 0, 0))  # Fully transparent
    res_rgba = ensure_three_channel_rgb(rgba)
    assert res_rgba.mode == "RGB"
    # Background should be solid white (255, 255, 255)
    pixel = res_rgba.getpixel((25, 25))
    assert pixel == (255, 255, 255)


def test_resize_image_deterministic():
    img = PIL.Image.new("RGB", (640, 480))
    resized = resize_image_deterministic(img, target_size=(224, 224), interpolation="bilinear")
    assert resized.size == (224, 224)


def test_image_to_normalized_tensor():
    img = PIL.Image.new("RGB", (224, 224), (128, 128, 128))
    tensor = image_to_normalized_tensor(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_compute_image_quality_metrics():
    # Flat uniform image has 0 variance (very low quality / blurred)
    flat = PIL.Image.new("RGB", (200, 200), (128, 128, 128))
    q_flat = compute_image_quality_metrics(flat)
    assert q_flat["laplacian_variance"] < 1.0
    assert q_flat["is_low_quality"] is True

    # High texture noise image has high variance
    np.random.seed(42)
    noise_arr = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    noise_img = PIL.Image.fromarray(noise_arr)
    q_noise = compute_image_quality_metrics(noise_img)
    assert q_noise["laplacian_variance"] > 100.0
    assert q_noise["is_low_quality"] is False
