"""
src/inference/predictor.py

Inference pipeline for the ai-character-detector.

Predictor class:
  1. Loads a saved checkpoint
  2. Validates and preprocesses the input image
  3. Optionally detects and crops a face region (OpenCV)
  4. Runs the model forward pass
  5. Returns prediction label + confidence score
  6. Optionally generates a Grad-CAM heatmap

Usage:
    predictor = Predictor(
        model_path="saved_models/best_model.pth",
        cfg=cfg,
    )

    result = predictor.predict("path/to/image.jpg")
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.transforms import denormalize, get_val_transforms
from src.models.classifier import build_model
from src.utils.face_detector import detect_and_crop_face
from src.utils.gradcam import generate_gradcam
from src.utils.image_validator import (
    ImageValidationError,
    validate_bytes,
)


IDX_TO_KEY = {
    0: "human",
    1: "cartoon",
    2: "ai_generated",
}

IDX_TO_DISPLAY = {
    0: "Real Human",
    1: "Cartoon / Anime",
    2: "AI Generated",
}


class Predictor:
    """
    End-to-end inference wrapper.

    Args:
        model_path: Path to the saved .pth checkpoint.
        cfg: Parsed config.yaml dictionary.
        device: Force a specific device. By default, CUDA is used
            when available; otherwise, CPU is used.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        cfg: dict,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = cfg

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.transform = get_val_transforms(self.cfg)
        self.model = self._load_model(Path(model_path))

        print(f"  [Predictor] Ready on {self.device}")

    def _load_model(self, model_path: Path) -> nn.Module:
        """
        Load the model architecture and checkpoint weights.
        """
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}\n"
                "Train the model first with: python train.py"
            )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint["model_state_dict"]

            if "cfg" in checkpoint:
                self.cfg = checkpoint["cfg"]
                self.transform = get_val_transforms(self.cfg)
        else:
            state_dict = checkpoint

        model = build_model(self.cfg)

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    def _preprocess(
        self,
        image_rgb: np.ndarray,
    ) -> torch.Tensor:
        """
        Apply validation transforms and return a tensor with shape
        (1, C, H, W).
        """
        if not isinstance(image_rgb, np.ndarray):
            raise TypeError(
                "image_rgb must be a NumPy array."
            )

        if image_rgb.ndim != 3:
            raise ValueError(
                "image_rgb must have shape (H, W, 3)."
            )

        if image_rgb.shape[2] != 3:
            raise ValueError(
                "image_rgb must contain exactly three RGB channels."
            )

        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(
                image_rgb,
                0,
                255,
            ).astype(np.uint8)

        transformed = self.transform(image=image_rgb)
        tensor = transformed["image"]

        if tensor.ndim != 3:
            raise ValueError(
                "The validation transform must return a "
                "(C, H, W) tensor."
            )

        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(
            device=self.device,
            dtype=torch.float32,
        )

        return tensor

    def _run_normal_inference(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """
        Run memory-efficient inference without tracking gradients.
        """
        self.model.eval()

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = F.softmax(
                logits,
                dim=1,
            ).squeeze(0)

        class_idx = int(
            probabilities.argmax(dim=0).item()
        )

        return probabilities, class_idx

    def _run_gradcam_inference(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, int, np.ndarray]:
        """
        Run prediction and generate Grad-CAM with gradient tracking.
        """
        self.model.eval()

        tensor = tensor.detach().clone()
        tensor.requires_grad_(True)

        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = self.model(tensor)

            probabilities = F.softmax(
                logits,
                dim=1,
            ).squeeze(0)

            class_idx = int(
                probabilities.argmax(dim=0).item()
            )

            original_image = denormalize(
                tensor.detach()
            )

            gradcam_overlay = generate_gradcam(
                model=self.model,
                image_tensor=tensor,
                target_class=class_idx,
                cfg=self.cfg,
                original_image=original_image,
            )

        self.model.zero_grad(set_to_none=True)

        probabilities = probabilities.detach()

        return (
            probabilities,
            class_idx,
            gradcam_overlay,
        )

    def predict_array(
        self,
        image_rgb: np.ndarray,
        return_gradcam: bool = False,
    ) -> dict:
        """
        Run inference on an RGB uint8 NumPy array.

        Args:
            image_rgb: Image with shape (H, W, 3), in RGB format.
            return_gradcam: Generate a Grad-CAM overlay when True.

        Returns:
            Dictionary containing the prediction, confidence,
            probabilities and optional Grad-CAM overlay.
        """
        if not isinstance(image_rgb, np.ndarray):
            raise TypeError(
                "image_rgb must be a NumPy array."
            )

        use_face_detection = self.cfg.get(
            "inference",
            {},
        ).get(
            "face_detection",
            True,
        )

        if use_face_detection:
            image_rgb = detect_and_crop_face(image_rgb)

        tensor = self._preprocess(image_rgb)

        gradcam_enabled = self.cfg.get(
            "inference",
            {},
        ).get(
            "gradcam",
            True,
        )

        should_generate_gradcam = (
            return_gradcam and gradcam_enabled
        )

        if should_generate_gradcam:
            (
                probabilities_tensor,
                class_idx,
                gradcam_overlay,
            ) = self._run_gradcam_inference(tensor)

        else:
            (
                probabilities_tensor,
                class_idx,
            ) = self._run_normal_inference(tensor)

            gradcam_overlay = None

        confidence = float(
            probabilities_tensor[class_idx].item()
        )

        probabilities = {
            IDX_TO_KEY[index]: round(
                float(
                    probabilities_tensor[index].item()
                ),
                4,
            )
            for index in range(len(IDX_TO_KEY))
        }

        return {
            "prediction": IDX_TO_DISPLAY[class_idx],
            "class_key": IDX_TO_KEY[class_idx],
            "class_idx": class_idx,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "gradcam": gradcam_overlay,
        }

    def predict(
        self,
        image_path: Union[str, Path],
        return_gradcam: bool = False,
    ) -> dict:
        """
        Run inference from a local image file.
        """
        image_path = Path(image_path)

        if not image_path.is_file():
            raise ImageValidationError(
                f"Image file not found: {image_path}"
            )

        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            raise ImageValidationError(
                f"Cannot read image: {image_path}"
            )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        return self.predict_array(
            image_rgb=image_rgb,
            return_gradcam=return_gradcam,
        )

    def predict_bytes(
        self,
        data: bytes,
        filename: str,
        return_gradcam: bool = False,
    ) -> dict:
        """
        Run inference from raw image bytes for the FastAPI endpoint.
        """
        allowed_extensions = set(
            self.cfg["api"]["allowed_extensions"]
        )

        max_file_size_mb = self.cfg["api"][
            "max_file_size_mb"
        ]

        pil_image = validate_bytes(
            data=data,
            filename=filename,
            allowed_ext=allowed_extensions,
            max_mb=max_file_size_mb,
        )

        image_rgb = np.asarray(
            pil_image.convert("RGB"),
            dtype=np.uint8,
        )

        return self.predict_array(
            image_rgb=image_rgb,
            return_gradcam=return_gradcam,
        )