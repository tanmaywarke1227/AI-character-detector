"""
src/api/server.py

FastAPI REST API for the ai-character-detector.

Endpoints:
  GET  /           → health check / welcome
  GET  /health     → JSON health status
  POST /predict    → image upload → prediction JSON
  POST /predict/gradcam → prediction + base64 Grad-CAM overlay

Run locally:
  uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000

Run via Docker:
  docker run -p 8000:8000 ai-character-detector
"""

import base64
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.inference.predictor import Predictor
from src.utils.image_validator import ImageValidationError


# ---------------------------------------------------------------------------
# Config & global predictor
# ---------------------------------------------------------------------------

def _load_cfg(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


_predictor: Optional[Predictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once on startup; release on shutdown."""
    global _predictor
    cfg = _load_cfg()
    model_path = cfg["paths"]["best_model"]
    try:
        _predictor = Predictor(model_path=model_path, cfg=cfg)
        print(f"  [API] Model loaded from {model_path}")
    except FileNotFoundError as e:
        print(f"  [API] WARNING: {e}")
        print("  [API] /predict will return 503 until a model is trained.")
    yield
    # Cleanup (nothing to do)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Character Detector",
    description=(
        "Classifies an uploaded image as Real Human, "
        "Cartoon / Anime, or AI Generated."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve the frontend
_frontend = Path("frontend")
if _frontend.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    prediction:    str
    class_key:     str
    confidence:    float
    probabilities: Dict[str, float]
    inference_ms:  float


class PredictionWithGradCAM(PredictionResponse):
    gradcam_base64: Optional[str] = None   # PNG encoded as base64 string


class HealthResponse(BaseModel):
    status:      str
    model_ready: bool
    version:     str = "1.0.0"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root():
    """Serve the frontend HTML page."""
    index = Path("frontend/index.html")
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(
        content="<h2>AI Character Detector API</h2>"
                "<p>POST an image to <code>/predict</code></p>"
                "<p>Frontend not found — place index.html in frontend/</p>",
        status_code=200,
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Return API and model health status."""
    return HealthResponse(
        status="ok",
        model_ready=_predictor is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(..., description="Image to classify")):
    """
    Classify an uploaded image.

    Returns the predicted class (Real Human / Cartoon Anime / AI Generated)
    and a confidence score.

    - **file**: JPEG, PNG, WebP, or BMP image (max 10 MB)
    """
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Train a model first with: python train.py",
        )

    data = await file.read()
    filename = file.filename or "upload.jpg"

    t0 = time.perf_counter()
    try:
        result = _predictor.predict_bytes(data, filename, return_gradcam=False)
    except ImageValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        prediction=result["prediction"],
        class_key=result["class_key"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        inference_ms=round(elapsed_ms, 2),
    )


@app.post("/predict/gradcam", response_model=PredictionWithGradCAM, tags=["Inference"])
async def predict_with_gradcam(
    file: UploadFile = File(..., description="Image to classify with Grad-CAM")
):
    """
    Classify an image AND return a Grad-CAM heatmap overlay.

    The heatmap is returned as a base64-encoded PNG string
    (`gradcam_base64`). Decode it on the client side to display.
    """
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )

    data = await file.read()
    filename = file.filename or "upload.jpg"

    t0 = time.perf_counter()
    try:
        result = _predictor.predict_bytes(data, filename, return_gradcam=True)
    except ImageValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Encode Grad-CAM overlay as base64 PNG
    gradcam_b64 = None
    if result.get("gradcam") is not None:
        from PIL import Image as PILImage
        import numpy as np
        pil_overlay = PILImage.fromarray(result["gradcam"].astype(np.uint8))
        buf = io.BytesIO()
        pil_overlay.save(buf, format="PNG")
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return PredictionWithGradCAM(
        prediction=result["prediction"],
        class_key=result["class_key"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        inference_ms=round(elapsed_ms, 2),
        gradcam_base64=gradcam_b64,
    )
