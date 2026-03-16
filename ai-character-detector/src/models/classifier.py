"""
src/models/classifier.py

Transfer-learning classifier for:
  0 → Real Human
  1 → Cartoon / Anime
  2 → AI Generated

Supported backbones:
  - resnet50
  - efficientnet_b0

Usage:
  model = build_model(cfg)
  model = model.to(device)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_B0_Weights,
)


# ---------------------------------------------------------------------------
# Custom classification head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Dropout → Linear → (optionally) BatchNorm head.
    Replaces the default fc / classifier layer of the backbone.
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> nn.Module:
    """
    Build a pretrained CNN with a custom 3-class head.

    Workflow:
      1. Load pretrained backbone (ImageNet weights).
      2. Freeze all backbone parameters.
      3. Attach custom ClassificationHead.
      4. Return model (caller is responsible for .to(device)).

    Unfreezing for fine-tuning is handled by the Trainer via
    `unfreeze_backbone()` after `cfg['model']['unfreeze_epoch']` epochs.

    Args:
        cfg : Parsed config.yaml dictionary.

    Returns:
        nn.Module ready for training.
    """
    backbone  = cfg["model"]["backbone"]
    num_cls   = cfg["model"]["num_classes"]
    pretrained = cfg["model"]["pretrained"]
    freeze    = cfg["model"]["freeze_backbone"]

    if backbone == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        if freeze:
            _freeze_all(model)
        model.fc = ClassificationHead(in_features, num_cls)

    elif backbone == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        if freeze:
            _freeze_all(model)
        model.classifier = ClassificationHead(in_features, num_cls)

    else:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. "
            "Choose 'resnet50' or 'efficientnet_b0'."
        )

    return model


# ---------------------------------------------------------------------------
# Fine-tuning helpers
# ---------------------------------------------------------------------------

def _freeze_all(model: nn.Module) -> None:
    """Freeze every parameter in the model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: nn.Module, cfg: dict) -> None:
    """
    Unfreeze the last N backbone blocks for fine-tuning.
    Called by the Trainer after `unfreeze_epoch` epochs.

    For ResNet50: unfreezes layer3 + layer4.
    For EfficientNet_B0: unfreezes features[-3:].
    """
    backbone = cfg["model"]["backbone"]

    if backbone == "resnet50":
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in ("layer3", "layer4", "fc")):
                param.requires_grad = True

    elif backbone == "efficientnet_b0":
        children = list(model.features.children())
        for block in children[-3:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [fine-tune] Trainable params: {trainable:,} / {total:,}")


# ---------------------------------------------------------------------------
# Inspection helper
# ---------------------------------------------------------------------------

def get_model_info(model: nn.Module) -> dict:
    """Return a dict with total / trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}
