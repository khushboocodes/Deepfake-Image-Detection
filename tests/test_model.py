"""
Unit and Integration Tests for Deep Learning Model Architecture Module.
"""

import os
import pytest
import torch
from src.model import (
    build_model,
    freeze_backbone,
    unfreeze_top_stages,
    get_parameter_groups,
    save_checkpoint,
    load_checkpoint,
    ForensicClassifier
)


def test_build_model_efficientnet():
    model = build_model("efficientnet_b0", pretrained=False)
    assert isinstance(model, ForensicClassifier)
    assert model.architecture == "efficientnet_b0"

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1)


def test_build_model_resnet18():
    model = build_model("resnet18", pretrained=False)
    assert isinstance(model, ForensicClassifier)
    assert model.architecture == "resnet18"

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1)


def test_freeze_and_unfreeze_backbone():
    model = build_model("efficientnet_b0", pretrained=False)
    freeze_backbone(model)

    # Backbone parameters should have requires_grad == False
    for p in model.features.parameters():
        assert p.requires_grad is False

    # Classifier head parameters should have requires_grad == True
    for p in model.classifier.parameters():
        assert p.requires_grad is True

    # Test unfreezing
    unfreeze_top_stages(model, num_stages=2)
    # At least some features parameters should now be trainable
    trainable_features = [p for p in model.features.parameters() if p.requires_grad]
    assert len(trainable_features) > 0


def test_checkpoint_serialization_cycle(tmp_path):
    model = build_model("efficientnet_b0", pretrained=False)
    chk_file = str(tmp_path / "test_model.pth")
    metrics = {"val_macro_f1": 0.925}

    save_checkpoint(model, None, epoch=5, metrics=metrics, filepath=chk_file)
    assert os.path.exists(chk_file)

    loaded_model = build_model("efficientnet_b0", pretrained=False)
    info = load_checkpoint(chk_file, loaded_model, device="cpu")
    assert info["epoch"] == 5
    assert info["metrics"]["val_macro_f1"] == 0.925
