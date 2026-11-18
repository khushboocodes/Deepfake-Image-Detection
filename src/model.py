"""
Deep Learning Model Architecture Module
Implements EfficientNet-B0 primary model and ResNet-18 fallback backbone
with custom forensic binary classification heads and transfer learning utilities.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights


def _init_head_weights(module: nn.Module) -> None:
    """Applies He / Kaiming normal initialization to intermediate layers and small normal to output layer."""
    if isinstance(module, nn.Linear):
        if module.out_features == 1:
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)


class ForensicClassifier(nn.Module):
    """
    Wrapper module combining a feature extraction backbone with a custom forensic classification head.
    """
    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.30
    ):
        super().__init__()
        self.architecture = architecture.lower()
        
        if self.architecture == "efficientnet_b0":
            try:
                weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
                base_model = models.efficientnet_b0(weights=weights)
            except Exception:
                # Offline fallback when downloading weights is blocked or unavailable
                base_model = models.efficientnet_b0(weights=None)
            
            # Backbone feature extractor
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            
            # Feature dimension from EfficientNet-B0 is 1280
            in_features = 1280
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 256),
                nn.SiLU(),
                nn.Dropout(p=dropout * 0.67),
                nn.Linear(256, 1)
            )
            self.classifier.apply(_init_head_weights)
            
        elif self.architecture == "resnet18":
            try:
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                base_model = models.resnet18(weights=weights)
            except Exception:
                base_model = models.resnet18(weights=None)
            
            # Extract convolutional layers up to average pooling
            self.features = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4
            )
            self.avgpool = base_model.avgpool
            
            # Feature dimension from ResNet-18 is 512
            in_features = 512
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(p=dropout * 0.67),
                nn.Linear(128, 1)
            )
            self.classifier.apply(_init_head_weights)
            
        else:
            raise ValueError(
                f"Unsupported architecture '{architecture}'. Supported: 'efficientnet_b0', 'resnet18'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass accepting input tensor [B, 3, 224, 224] and returning scalar logits [B, 1].
        """
        feats = self.features(x)
        pooled = self.avgpool(feats)
        logits = self.classifier(pooled)
        return logits


def build_model(
    architecture: str = "efficientnet_b0",
    pretrained: bool = True,
    dropout: float = 0.30
) -> ForensicClassifier:
    """
    Constructs and returns the ForensicClassifier model instance.
    """
    return ForensicClassifier(
        architecture=architecture,
        pretrained=pretrained,
        dropout=dropout
    )


def freeze_backbone(model: ForensicClassifier) -> None:
    """
    Freezes all feature extractor parameters in the backbone for Stage 1 warmup training.
    Only classifier head parameters retain requires_grad = True.
    """
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.avgpool.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_top_stages(model: ForensicClassifier, num_stages: int = 2) -> None:
    """
    Unfreezes top convolutional feature blocks for Stage 2 fine-tuning.
    """
    if model.architecture == "efficientnet_b0":
        # Stages 6 and 7 of EfficientNet-B0 features
        total_blocks = len(model.features)  # Typically 9 (stem + 7 MBConv stages + conv_head)
        start_idx = max(0, total_blocks - num_stages - 1)
        for i in range(start_idx, total_blocks):
            for param in model.features[i].parameters():
                param.requires_grad = True
                
    elif model.architecture == "resnet18":
        # In ResNet-18, features[6] is layer3 and features[7] is layer4
        for param in model.features[7].parameters():  # layer4
            param.requires_grad = True
        if num_stages >= 2:
            for param in model.features[6].parameters():  # layer3
                param.requires_grad = True


def unfreeze_all_backbone(model: ForensicClassifier) -> None:
    """
    Unfreezes all parameters across the entire backbone for full end-to-end fine-tuning.
    """
    for param in model.parameters():
        param.requires_grad = True


def get_parameter_groups(
    model: ForensicClassifier,
    backbone_lr: float = 3e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4
) -> List[Dict[str, Any]]:
    """
    Constructs differential parameter groups: lower LR for pretrained backbone, higher LR for custom head.
    """
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    groups = []
    if backbone_params:
        groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay
        })
    if head_params:
        groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay
        })
        
    return groups


def save_checkpoint(
    model: ForensicClassifier,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, Any],
    filepath: str
) -> None:
    """
    Serializes model weights, optimizer state, epoch index, and evaluation telemetry.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "architecture": model.architecture,
        "torch_version": str(torch.__version__)
    }
    torch.save(payload, filepath)


def load_checkpoint(
    filepath: str,
    model: ForensicClassifier,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Loads saved checkpoint weights into model and sets model to evaluation mode.
    
    Returns:
        dict: Checkpoint metadata dictionary (epoch, metrics, architecture).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model checkpoint not found at: '{filepath}'")
        
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(filepath, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "architecture": checkpoint.get("architecture", model.architecture),
        "torch_version": checkpoint.get("torch_version", "")
    }
