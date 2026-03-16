"""
src/data/dataset.py

Custom PyTorch Dataset for the ai-character-detector project.

Expected folder layout:
  data/processed/
    train/
      human/         *.jpg / *.png
      cartoon/       *.jpg / *.png
      ai_generated/  *.jpg / *.png
    val/   (same structure)
    test/  (same structure)
"""

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage
from torch.utils.data import Dataset


# Map class folder name → integer label
CLASS_TO_IDX = {
    "human": 0,
    "cartoon": 1,
    "ai_generated": 2,
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

DISPLAY_NAMES = {
    "human": "Real Human",
    "cartoon": "Cartoon / Anime",
    "ai_generated": "AI Generated",
}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _scan_images(root: Path, classes: List[str]) -> List[Tuple[Path, int]]:
    """Recursively collect (image_path, label) pairs under root/class_name/."""
    samples = []
    for cls in classes:
        cls_dir = root / cls
        if not cls_dir.exists():
            continue
        label = CLASS_TO_IDX[cls]
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in SUPPORTED_EXTS:
                samples.append((img_path, label))
    return samples


class CharacterDataset(Dataset):
    """
    Loads images from a split folder and applies Albumentations transforms.

    Args:
        root        : Path to split root, e.g. 'data/processed/train'
        transform   : Albumentations Compose pipeline (or None)
        classes     : List of class names to include (default: all three)
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        classes: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.classes = classes or list(CLASS_TO_IDX.keys())
        self.samples: List[Tuple[Path, int]] = _scan_images(self.root, self.classes)

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root}. "
                "Did you run download_dataset.py first?"
            )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]

        # Read with OpenCV (BGR) → convert to RGB numpy array
        img = cv2.imread(str(img_path))
        if img is None:
            # Fallback: try PIL
            try:
                img = np.array(PILImage.open(img_path).convert("RGB"))
            except Exception:
                # Return a black image rather than crashing the DataLoader
                img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transforms (Albumentations expects numpy uint8 H×W×C)
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]  # now a torch.Tensor C×H×W float32

        return img, label

    # ------------------------------------------------------------------
    def class_counts(self) -> dict:
        """Return {class_name: count} for this split."""
        counts = {c: 0 for c in self.classes}
        for _, label in self.samples:
            counts[IDX_TO_CLASS[label]] += 1
        return counts

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        counts = self.class_counts()
        return (
            f"CharacterDataset(root={self.root}, "
            f"n={len(self)}, counts={counts})"
        )
