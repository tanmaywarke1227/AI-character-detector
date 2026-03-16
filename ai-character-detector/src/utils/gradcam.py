"""
src/utils/gradcam.py

Grad-CAM visualisation using the pytorch-grad-cam library.

Returns a heatmap overlaid on the original image as a numpy uint8 array.

Usage:
    heatmap = generate_gradcam(model, image_tensor, target_class, cfg)
    # heatmap is a (H, W, 3) numpy array in BGR for OpenCV, or RGB for PIL
"""

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from pytorch_grad_cam import GradCAM  # type: ignore
    from pytorch_grad_cam.utils.image import show_cam_on_image  # type: ignore
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # type: ignore
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


def _get_target_layers(model: nn.Module, backbone: str) -> list:
    """
    Return the target conv layers for Grad-CAM.

    For ResNet50      → last residual block of layer4
    For EfficientNet  → last inverted residual block of features
    """
    if backbone == "resnet50":
        return [model.layer4[-1]]
    elif backbone == "efficientnet_b0":
        return [model.features[-1]]
    else:
        # Fallback: try to find the last Conv2d
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if conv_layers:
            return [conv_layers[-1]]
        raise ValueError(f"Cannot auto-detect target layer for backbone: {backbone}")


def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: Optional[int],
    cfg: dict,
    original_image: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Generate a Grad-CAM heatmap for the given image tensor.

    Args:
        model          : The trained nn.Module.
        image_tensor   : Preprocessed tensor of shape (1, C, H, W).
        target_class   : Class index to explain. If None, uses predicted class.
        cfg            : Parsed config.yaml dictionary.
        original_image : Denormalized RGB numpy uint8 (H, W, 3) for overlay.
                         If None, the overlay uses a blank white canvas.

    Returns:
        RGB uint8 numpy array (H, W, 3) with the heatmap overlaid,
        or None if pytorch-grad-cam is not installed.
    """
    if not GRADCAM_AVAILABLE:
        print(
            "  [gradcam] pytorch-grad-cam not installed. "
            "Run: pip install grad-cam"
        )
        return None

    backbone = cfg["model"]["backbone"]
    size     = cfg["data"]["image_size"]

    try:
        target_layers = _get_target_layers(model, backbone)
    except ValueError as e:
        print(f"  [gradcam] {e}")
        return None

    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=targets,
        )
        grayscale_cam = grayscale_cam[0]  # (H, W) float32 in [0, 1]

    # Prepare background image for overlay
    if original_image is not None:
        # Resize to match cam output
        bg = cv2.resize(original_image, (size, size))
        bg = bg.astype(np.float32) / 255.0
    else:
        bg = np.ones((size, size, 3), dtype=np.float32)

    overlay = show_cam_on_image(bg, grayscale_cam, use_rgb=True)
    return overlay  # (H, W, 3) uint8


def save_gradcam(overlay: np.ndarray, save_path: str) -> None:
    """Save a Grad-CAM overlay to disk (converts RGB → BGR for OpenCV)."""
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, bgr)
    print(f"  [gradcam] Saved → {save_path}")
