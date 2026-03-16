# AI Character Detector

A production-grade deep learning system that classifies any image as:

| Label | Description |
|---|---|
| 👤 **Real Human** | Photographs of real people |
| 🎨 **Cartoon / Anime** | Hand-drawn or 2-D animated characters |
| 🤖 **AI Generated** | Images synthesised by diffusion models (Stable Diffusion, MidJourney, DALL-E, etc.) |

Built with PyTorch, FastAPI, and Albumentations. Targets **90–95% accuracy** with transfer learning on ~100k images.

---

## Project Structure

```
ai-character-detector/
├── data/
│   ├── raw/
│   │   ├── human/           ← CelebA images
│   │   ├── cartoon/         ← Anime / Danbooru images
│   │   └── ai_generated/    ← Stable Diffusion / MidJourney images
│   └── processed/
│       ├── train/  val/  test/
├── frontend/
│   └── index.html           ← Upload UI
├── logs/                    ← TensorBoard + training curves
├── notebooks/               ← Exploration notebooks
├── saved_models/
│   ├── best_model.pth
│   └── last_model.pth
├── src/
│   ├── data/
│   │   ├── dataset.py       ← PyTorch Dataset
│   │   ├── dataloader.py    ← DataLoader factory
│   │   └── transforms.py    ← Albumentations pipelines
│   ├── models/
│   │   └── classifier.py    ← ResNet50 / EfficientNet builder
│   ├── training/
│   │   ├── trainer.py       ← Full train/val loop + TensorBoard
│   │   ├── early_stopping.py
│   │   └── evaluator.py     ← Metrics + confusion matrix
│   ├── inference/
│   │   └── predictor.py     ← End-to-end inference class
│   ├── api/
│   │   └── server.py        ← FastAPI REST server
│   └── utils/
│       ├── gradcam.py       ← Grad-CAM heatmaps
│       ├── image_validator.py
│       └── face_detector.py ← OpenCV face crop (optional)
├── config.yaml              ← All hyperparameters
├── train.py                 ← Training entry point
├── predict.py               ← CLI inference
├── download_dataset.py      ← Dataset downloader + splitter
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/ai-character-detector.git
cd ai-character-detector

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Download CelebA (human) + Danbooru (cartoon) and split 80/10/10
python download_dataset.py --source all --max-images 30000

# AI-generated images must be added manually:
#   → Place JPG/PNG files in:  data/raw/ai_generated/
#   → Then re-run the splitter:
python download_dataset.py --split-only
```

**Dataset size guide:**

| Class | Recommended | Minimum |
|---|---|---|
| human | 30,000 | 5,000 |
| cartoon | 30,000 | 5,000 |
| ai_generated | 30,000 | 5,000 |

### 3. Train

```bash
# Train with ResNet50 (default)
python train.py

# Train with EfficientNet-B0
python train.py --backbone efficientnet_b0

# Smoke test (2 epochs, 2 batches each)
python train.py --dry-run

# Resume from checkpoint
python train.py --resume saved_models/last_model.pth
```

Watch live metrics:
```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```

### 4. Evaluate

```bash
python train.py --eval-only
```

### 5. Predict

```bash
# Single image
python predict.py --image path/to/photo.jpg

# With Grad-CAM heatmap
python predict.py --image photo.jpg --gradcam

# Entire folder
python predict.py --folder data/test_images/
```

### 6. Start the API

```bash
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

Open the UI: **http://localhost:8000**

API docs: **http://localhost:8000/docs**

---

## API Reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@photo.jpg"
```

**Response:**
```json
{
  "prediction":    "Real Human",
  "class_key":     "human",
  "confidence":    0.9421,
  "probabilities": {
    "human":        0.9421,
    "cartoon":      0.0391,
    "ai_generated": 0.0188
  },
  "inference_ms": 18.4
}
```

### `POST /predict/gradcam`

Same as above plus a `gradcam_base64` field containing a base64-encoded PNG heatmap.

### `GET /health`

```json
{ "status": "ok", "model_ready": true, "version": "1.0.0" }
```

---

## Docker

```bash
# Build
docker build -t ai-character-detector .

# Run (mount your trained model)
docker run -p 8000:8000 \
  -v $(pwd)/saved_models:/app/saved_models \
  ai-character-detector

# With GPU
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/saved_models:/app/saved_models \
  ai-character-detector
```

---

## Configuration

All hyperparameters live in `config.yaml`. Key settings:

```yaml
model:
  backbone: resnet50          # or efficientnet_b0
  freeze_backbone: true       # train head only at first
  unfreeze_epoch: 5           # then fine-tune last blocks

training:
  batch_size: 32
  epochs: 30
  learning_rate: 0.001
  early_stopping_patience: 7
  scheduler: cosine
  label_smoothing: 0.1
```

---

## Model Architecture

```
Input image (224×224×3)
        ↓
Pretrained backbone (ResNet50 or EfficientNet-B0)
  Phase 1: backbone frozen, only head trains  (epochs 1–4)
  Phase 2: last 2 blocks + head fine-tuned    (epoch 5+)
        ↓
ClassificationHead:
  BatchNorm1d → Dropout(0.3) → Linear(in→512) → ReLU → Dropout(0.15) → Linear(512→3)
        ↓
Softmax → [P(human), P(cartoon), P(ai_generated)]
```

Training uses **label-smoothed cross-entropy** + **AdamW** + **CosineAnnealingLR**.

---

## Expected Performance

With ~90k images (30k per class) and ResNet50:

| Metric | Expected |
|---|---|
| Accuracy | 90–95% |
| Weighted F1 | 0.90–0.95 |
| Inference (CPU) | ~50ms |
| Inference (GPU) | ~10ms |

---

## Acknowledgements

- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) — Liu et al., ICCV 2015
- [Danbooru Dataset](https://danbooru.donmai.us/) — via HuggingFace
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — Jacob Gildenblat
- [Albumentations](https://albumentations.ai/) — Buslaev et al.
