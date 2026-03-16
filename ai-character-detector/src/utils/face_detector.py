"""
src/utils/face_detector.py

Optional face detection using OpenCV's Haar Cascade classifier.

If a face is detected in the image, the face region is cropped and
returned for classification. This can improve accuracy for portraits.

If no face is detected, the original image is returned unchanged.

Usage:
    from src.utils.face_detector import detect_and_crop_face
    cropped = detect_and_crop_face(image_array)
"""

from typing import Optional
import cv2
import numpy as np
from pathlib import Path


# OpenCV ships the cascade XML — resolve path relative to cv2 package
_CASCADE_PATH = str(
    Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"
)


def detect_and_crop_face(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_face_size: int = 80,
    padding: float = 0.2,
) -> np.ndarray:
    """
    Detect the largest face in an RGB image and return a padded crop.

    Args:
        image          : RGB numpy uint8 array (H, W, 3).
        scale_factor   : How much the image size is reduced at each scale.
        min_neighbors  : How many neighbors each candidate rect should have.
        min_face_size  : Minimum face dimension in pixels.
        padding        : Fractional padding around the detected face.

    Returns:
        Cropped face region (RGB uint8), or the original image if no
        face is found.
    """
    # Load cascade (cached internally by OpenCV after first call)
    cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    if cascade.empty():
        # Cascade file missing — return original image silently
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = image.shape[:2]

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face_size, min_face_size),
    )

    if not isinstance(faces, np.ndarray) or len(faces) == 0:
        return image  # No face found — use full image

    # Pick the largest face (by area)
    areas  = [fw * fh for (_, _, fw, fh) in faces]
    x, y, fw, fh = faces[int(np.argmax(areas))]

    # Add padding
    pad_x = int(fw * padding)
    pad_y = int(fh * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + fw + pad_x)
    y2 = min(h, y + fh + pad_y)

    return image[y1:y2, x1:x2]


def has_face(image: np.ndarray) -> bool:
    """Return True if at least one face is detected in the image."""
    result = detect_and_crop_face(image)
    return result.shape != image.shape
