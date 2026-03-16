"""
src/data/transforms.py

Albumentations-based image transformation pipelines.
  - get_train_transforms : heavy augmentation for training
  - get_val_transforms   : resize + normalize only for val/test
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet statistics (used because backbone is pretrained on ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(cfg: dict) -> A.Compose:
    """
    Return augmentation pipeline for the training split.

    Augmentations are controlled by values in cfg['augmentation'].
    All transforms operate on numpy uint8 images (H x W x C).
    The final ToTensorV2 converts to a PyTorch float32 tensor (C x H x W).
    """
    aug = cfg.get("augmentation", {})
    size = cfg["data"]["image_size"]

    return A.Compose([
        # --- Geometric ---
        A.Resize(size, size),
        A.HorizontalFlip(p=aug.get("horizontal_flip", 0.5)),
        A.Rotate(limit=aug.get("rotation_limit", 15), p=0.4),
        A.RandomResizedCrop(
            size=(size, size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=aug.get("random_crop", 0.3),
        ),

        # --- Colour ---
        A.ColorJitter(
            brightness=aug.get("brightness", 0.2),
            contrast=aug.get("contrast", 0.2),
            saturation=aug.get("saturation", 0.2),
            hue=0.05,
            p=aug.get("color_jitter", 0.4),
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=aug.get("blur", 0.2)),
        A.ToGray(p=0.05),                       # Rarely convert to grayscale

        # --- Noise ---
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.15),

        # --- Final ---
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(cfg: dict) -> A.Compose:
    """
    Return transformation pipeline for validation and test splits.
    No augmentation — just resize and normalize.
    """
    size = cfg["data"]["image_size"]
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def denormalize(tensor):
    """
    Reverse ImageNet normalization for visualization purposes.
    Input : torch.Tensor of shape (C, H, W) or (B, C, H, W)
    Output: numpy uint8 array (H, W, C) in [0, 255]
    """
    import numpy as np
    import torch

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    if tensor.dim() == 4:
        tensor = tensor[0]

    img = tensor.cpu().float() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)
