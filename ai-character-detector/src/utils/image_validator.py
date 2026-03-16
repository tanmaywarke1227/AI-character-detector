"""
src/utils/image_validator.py

Validates an uploaded image before running inference.

Checks:
  • File extension is in the allowed list
  • File size does not exceed the configured limit
  • File is a valid, non-corrupted image (PIL can open it)
  • Image has at least one colour channel (not just metadata)
"""

from pathlib import Path
from typing import Union

from PIL import Image as PILImage


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
DEFAULT_MAX_MB = 10


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation."""
    pass


def validate_image(
    path: Union[str, Path],
    allowed_ext: set = ALLOWED_EXTENSIONS,
    max_mb: float = DEFAULT_MAX_MB,
) -> PILImage.Image:
    """
    Validate and return a PIL Image object.

    Args:
        path        : Path to the image file.
        allowed_ext : Set of allowed file extensions (lowercase, without dot).
        max_mb      : Maximum allowed file size in megabytes.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ImageValidationError : If any check fails.
    """
    path = Path(path)

    # 1. File exists
    if not path.exists():
        raise ImageValidationError(f"File not found: {path}")

    # 2. Extension check
    ext = path.suffix.lstrip(".").lower()
    if ext not in allowed_ext:
        raise ImageValidationError(
            f"Unsupported file type '.{ext}'. "
            f"Allowed: {sorted(allowed_ext)}"
        )

    # 3. File size check
    size_mb = path.stat().st_size / (1024 ** 2)
    if size_mb > max_mb:
        raise ImageValidationError(
            f"File too large ({size_mb:.1f} MB). Maximum allowed: {max_mb} MB."
        )

    # 4. Open and verify the image
    try:
        img = PILImage.open(path)
        img.verify()          # Catches truncated / corrupted files
    except Exception as exc:
        raise ImageValidationError(f"Corrupted or unreadable image: {exc}") from exc

    # Re-open after verify() (PIL closes the file after verify)
    try:
        img = PILImage.open(path).convert("RGB")
    except Exception as exc:
        raise ImageValidationError(f"Cannot decode image: {exc}") from exc

    # 5. Minimum dimensions
    w, h = img.size
    if w < 32 or h < 32:
        raise ImageValidationError(
            f"Image too small ({w}×{h}px). Minimum: 32×32."
        )

    return img


def validate_bytes(
    data: bytes,
    filename: str,
    allowed_ext: set = ALLOWED_EXTENSIONS,
    max_mb: float = DEFAULT_MAX_MB,
) -> PILImage.Image:
    """
    Validate raw bytes (as received from FastAPI UploadFile).

    Args:
        data        : Raw image bytes.
        filename    : Original filename (used to check extension).
        allowed_ext : Set of allowed extensions.
        max_mb      : Maximum size in megabytes.

    Returns:
        PIL Image in RGB mode.
    """
    import io

    # Extension check
    ext = Path(filename).suffix.lstrip(".").lower()
    if ext not in allowed_ext:
        raise ImageValidationError(
            f"Unsupported file type '.{ext}'. Allowed: {sorted(allowed_ext)}"
        )

    # Size check
    size_mb = len(data) / (1024 ** 2)
    if size_mb > max_mb:
        raise ImageValidationError(
            f"File too large ({size_mb:.1f} MB). Maximum: {max_mb} MB."
        )

    # Decode
    try:
        img = PILImage.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
    except Exception as exc:
        raise ImageValidationError(f"Cannot decode image: {exc}") from exc

    if w < 32 or h < 32:
        raise ImageValidationError(
            f"Image too small ({w}×{h}px). Minimum: 32×32."
        )

    return img
