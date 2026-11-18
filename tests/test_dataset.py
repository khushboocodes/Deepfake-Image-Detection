"""
Unit and Integration Tests for Dataset Engineering and DataLoader Module.
"""

import os
import pytest
import numpy as np
import PIL.Image
import torch
from src.dataset import (
    compute_phash_numpy,
    generate_splits,
    DeepfakeDataset,
    create_dataloader
)


def test_compute_phash_numpy():
    img1 = PIL.Image.new("RGB", (100, 100), (200, 100, 50))
    h1 = compute_phash_numpy(img1)
    assert isinstance(h1, str)
    assert len(h1) == 16

    # Similar images should produce identical or close hashes
    img2 = PIL.Image.new("RGB", (100, 100), (202, 101, 51))
    h2 = compute_phash_numpy(img2)
    assert h1 == h2


def test_generate_splits_stratification():
    # 20 samples: 10 real, 10 fake across two generators
    mock_manifest = [
        {"asset_id": f"r_{i}", "label": 0, "generator_family": "ImageNet"} for i in range(10)
    ] + [
        {"asset_id": f"f_{i}", "label": 1, "generator_family": "SD" if i % 2 == 0 else "MJ"} for i in range(10)
    ]

    train_s, val_s, test_s = generate_splits(
        mock_manifest,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    assert len(train_s) + len(val_s) + len(test_s) == 20
    # Zero leakage between splits
    train_ids = {x["asset_id"] for x in train_s}
    val_ids = {x["asset_id"] for x in val_s}
    test_ids = {x["asset_id"] for x in test_s}

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0


def test_deepfake_dataset_dataloader(tmp_path):
    # Create two dummy images on disk
    p1 = str(tmp_path / "img1.jpg")
    p2 = str(tmp_path / "img2.jpg")
    PIL.Image.new("RGB", (128, 128), (100, 150, 200)).save(p1)
    PIL.Image.new("RGB", (128, 128), (200, 150, 100)).save(p2)

    manifest = [
        {"asset_id": "a1", "relative_path": p1, "label": 0},
        {"asset_id": "a2", "relative_path": p2, "label": 1}
    ]

    dataset = DeepfakeDataset(manifest, is_training=False, img_size=(224, 224))
    assert len(dataset) == 2

    tensor, label, asset_id = dataset[0]
    assert tensor.shape == (3, 224, 224)
    assert label.shape == (1,)
    assert label.item() == 0.0
    assert asset_id == "a1"

    loader = create_dataloader(manifest, batch_size=2, is_training=False)
    batch_tensors, batch_labels, batch_ids = next(iter(loader))
    assert batch_tensors.shape == (2, 3, 224, 224)
    assert batch_labels.shape == (2, 1)
