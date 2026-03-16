# ============================================================
# Dockerfile — AI Character Detector
# ============================================================
#
# Build:
#   docker build -t ai-character-detector .
#
# Run (with a pre-trained model mounted):
#   docker run -p 8000:8000 \
#     -v $(pwd)/saved_models:/app/saved_models \
#     ai-character-detector
#
# Run with GPU (requires nvidia-container-toolkit):
#   docker run --gpus all -p 8000:8000 \
#     -v $(pwd)/saved_models:/app/saved_models \
#     ai-character-detector
# ============================================================

FROM python:3.10-slim AS base

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----
WORKDIR /app
COPY requirements.txt .

# Install CPU-only PyTorch first (smaller image), then the rest
RUN pip install --no-cache-dir \
        torch==2.1.0 torchvision==0.16.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ---- Application code ----
COPY . .

# Create directories that must exist at runtime
RUN mkdir -p saved_models logs data/processed frontend

# ---- Non-root user (security best practice) ----
RUN useradd --create-home appuser
RUN chown -R appuser:appuser /app
USER appuser

# ---- Runtime ----
EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

CMD ["uvicorn", "src.api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
