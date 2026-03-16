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
    predictor = Predictor(model_path="saved_models/best_model.pth", cfg=cfg)
    result = predictor.predict("path/to/image.jpg")
    # {'prediction': 'Real Human', 'class_key': 'human',
    #  'confidence': 0.94, 'probabilities': {...}, 'gradcam': <array|None>}
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.transforms import get_val_transforms, denormalize
from src.models.classifier import build_model
from src.utils.image_validator import validate_bytes, ImageValidationError
from src.utils.face_detector import detect_and_crop_face
from src.utils.gradcam import generate_gradcam


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

IDX_TO_KEY = {0: "human", 1: "cartoon", 2: "ai_generated"}
IDX_TO_DISPLAY = {
    0: "Real Human",
    1: "Cartoon / Anime",
    2: "AI Generated",
}


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """
    End-to-end inference wrapper.

    Args:
        model_path : Path to the saved .pth checkpoint.
        cfg        : Parsed config.yaml dictionary.
        device     : Force a specific device (default: auto-detect).
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        cfg: dict,
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.transform = get_val_transforms(cfg)
        self.model = self._load_model(Path(model_path))
        print(f"  [Predictor] Ready on {self.device}")

    # ------------------------------------------------------------------
    def _load_model(self, model_path: Path) -> nn.Module:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}\n"
                "Train the model first with: python train.py"
            )

        checkpoint = torch.load(model_path, map_location=self.device)

        # Support checkpoints saved by Trainer (dict) or plain state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            # Override cfg from checkpoint if available
            if "cfg" in checkpoint:
                self.cfg = checkpoint["cfg"]
        else:
            state_dict = checkpoint

        model = build_model(self.cfg)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Apply val transforms and return a (1, C, H, W) tensor."""
        augmented = self.transform(image=image_rgb)
        tensor = augmented["image"].unsqueeze(0)  # (1, C, H, W)
        return tensor.to(self.device)

    # ------------------------------------------------------------------
    def predict_array(
        self,
        image_rgb: np.ndarray,
        return_gradcam: bool = False,
    ) -> dict:
        """
        Run inference on a numpy RGB uint8 array.

        Args:
            image_rgb     : (H, W, 3) numpy array in RGB, uint8.
            return_gradcam: Whether to compute Grad-CAM overlay.

        Returns:
            Dict with keys:
              prediction   : str  — human-readable class name
              class_key    : str  — internal key  ('human'|'cartoon'|'ai_generated')
              class_idx    : int  — integer label (0|1|2)
              confidence   : float in [0, 1]
              probabilities: dict {class_key: probability}
              gradcam      : np.ndarray | None
        """
        # Optional face crop
        use_face = self.cfg.get("inference", {}).get("face_detection", True)
        if use_face:
            image_rgb = detect_and_crop_face(image_rgb)

        tensor = self._preprocess(image_rgb)

        with torch.no_grad():
            logits = self.model(tensor)                       # (1, 3)
            probs  = F.softmax(logits, dim=1).squeeze()      # (3,)

        class_idx  = int(probs.argmax().item())
        confidence = float(probs[class_idx].item())

        probabilities = {
            IDX_TO_KEY[i]: round(float(probs[i].item()), 4)
            for i in range(len(IDX_TO_KEY))
        }

        # Grad-CAM
        gradcam_overlay = None
        use_gradcam = self.cfg.get("inference", {}).get("gradcam", True)
        if return_gradcam and use_gradcam:
            orig_vis = denormalize(tensor)   # (H, W, 3) uint8 RGB
            gradcam_overlay = generate_gradcam(
                model=self.model,
                image_tensor=tensor,
                target_class=class_idx,
                cfg=self.cfg,
                original_image=orig_vis,
            )

        return {
            "prediction":    IDX_TO_DISPLAY[class_idx],
            "class_key":     IDX_TO_KEY[class_idx],
            "class_idx":     class_idx,
            "confidence":    round(confidence, 4),
            "probabilities": probabilities,
            "gradcam":       gradcam_overlay,
        }

    # ------------------------------------------------------------------
    def predict(
        self,
        image_path: Union[str, Path],
        return_gradcam: bool = False,
    ) -> dict:
        """
        Run inference from a file path.

        Validates the file before processing.
        """
        image_path = Path(image_path)

        # Read with OpenCV (more robust than PIL for varied formats)
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ImageValidationError(f"Cannot read image: {image_path}")
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        return self.predict_array(image_rgb, return_gradcam=return_gradcam)

    # ------------------------------------------------------------------
    def predict_bytes(
        self,
        data: bytes,
        filename: str,
        return_gradcam: bool = False,
    ) -> dict:
        """
        Run inference from raw bytes (used by the FastAPI endpoint).

        Validates file extension and size, then decodes.
        """
        allowed = set(self.cfg["api"]["allowed_extensions"])
        max_mb  = self.cfg["api"]["max_file_size_mb"]

        # validate_bytes returns a PIL Image in RGB
        pil_img = validate_bytes(data, filename, allowed_ext=allowed, max_mb=max_mb)
        image_rgb = np.array(pil_img)   # (H, W, 3) uint8 RGB

        return self.predict_array(image_rgb, return_gradcam=return_gradcam)
