"""
predict.py

Command-line inference script for the ai-character-detector.

Usage:
    # Predict a single image
    python predict.py --image path/to/image.jpg

    # Save Grad-CAM heatmap alongside result
    python predict.py --image path/to/image.jpg --gradcam

    # Save Grad-CAM to a custom path
    python predict.py --image photo.png --gradcam --gradcam-out heatmap.png

    # Predict all images in a folder (prints a table)
    python predict.py --folder data/test_images/

    # Use a different model checkpoint
    python predict.py --image photo.jpg --model saved_models/last_model.pth
"""

import argparse
import sys
import time
from pathlib import Path

import yaml


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CLASS_DISPLAY = {
    "human":       "Real Human",
    "cartoon":     "Cartoon / Anime",
    "ai_generated":"AI Generated",
}

EMOJI = {
    "human": "👤",
    "cartoon": "🎨",
    "ai_generated": "🤖",
}


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classify an image as Real Human / Cartoon / AI Generated",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Path to a single image file")
    group.add_argument("--folder", help="Path to a folder of images")

    p.add_argument(
        "--model",
        default=None,
        help="Path to model checkpoint (default: from config.yaml)",
    )
    p.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate and save a Grad-CAM heatmap",
    )
    p.add_argument(
        "--gradcam-out",
        default=None,
        help="Output path for Grad-CAM PNG (default: <image_stem>_gradcam.png)",
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Show top-k class probabilities",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single-image prediction
# ---------------------------------------------------------------------------

def predict_single(
    predictor,
    image_path: Path,
    args: argparse.Namespace,
) -> dict:
    """Run prediction on one image; print result; optionally save Grad-CAM."""
    t0 = time.perf_counter()
    result = predictor.predict(str(image_path), return_gradcam=args.gradcam)
    elapsed = (time.perf_counter() - t0) * 1000

    key  = result["class_key"]
    conf = result["confidence"]
    icon = EMOJI.get(key, "")

    print(f"\n  Image     : {image_path.name}")
    print(f"  Prediction: {icon}  {result['prediction']}")
    print(f"  Confidence: {conf:.1%}")
    print(f"  Inference : {elapsed:.1f} ms")
    print()

    # Top-k probabilities
    probs_sorted = sorted(
        result["probabilities"].items(), key=lambda x: x[1], reverse=True
    )
    print("  Class probabilities:")
    for cls_key, prob in probs_sorted[: args.top_k]:
        bar_len = int(prob * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {CLASS_DISPLAY[cls_key]:<20s}  {bar}  {prob:.1%}")
    print()

    # Save Grad-CAM
    if args.gradcam and result.get("gradcam") is not None:
        from src.utils.gradcam import save_gradcam

        if args.gradcam_out:
            out_path = args.gradcam_out
        else:
            out_path = str(image_path.parent / f"{image_path.stem}_gradcam.png")

        save_gradcam(result["gradcam"], out_path)

    return result


# ---------------------------------------------------------------------------
# Folder prediction
# ---------------------------------------------------------------------------

def predict_folder(predictor, folder: Path, args: argparse.Namespace) -> None:
    """Run prediction on every image in a folder and print a summary table."""
    images = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )

    if not images:
        print(f"  No images found in {folder}")
        return

    print(f"\n  Found {len(images)} images in {folder}\n")
    print(f"  {'Image':<35s}  {'Prediction':<20s}  {'Confidence':>10s}")
    print("  " + "-" * 72)

    counts = {"human": 0, "cartoon": 0, "ai_generated": 0}
    errors = []

    for img_path in images:
        try:
            result = predictor.predict(str(img_path), return_gradcam=False)
            key    = result["class_key"]
            conf   = result["confidence"]
            icon   = EMOJI.get(key, "")
            counts[key] += 1
            label  = f"{icon} {result['prediction']}"
            name   = img_path.name[:34]
            print(f"  {name:<35s}  {label:<20s}  {conf:>9.1%}")
        except Exception as e:
            errors.append((img_path.name, str(e)))
            print(f"  {img_path.name:<35s}  ERROR: {e}")

    # Summary
    total = len(images) - len(errors)
    print("\n  " + "-" * 72)
    print(f"\n  Summary ({total} images classified):")
    for key, label in CLASS_DISPLAY.items():
        pct = counts[key] / total if total else 0
        print(f"    {label:<20s}: {counts[key]:>4d}  ({pct:.0%})")
    if errors:
        print(f"\n  {len(errors)} image(s) failed — see errors above.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    # Resolve model path
    model_path = args.model or cfg["paths"]["best_model"]

    # Import here (avoid loading torch for --help)
    from src.inference.predictor import Predictor

    print("\n  AI Character Detector — Inference")
    print(f"  Model: {model_path}")

    try:
        predictor = Predictor(model_path=model_path, cfg=cfg)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"\n  ERROR: File not found: {image_path}")
            sys.exit(1)
        predict_single(predictor, image_path, args)

    elif args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"\n  ERROR: Not a directory: {folder}")
            sys.exit(1)
        predict_folder(predictor, folder, args)


if __name__ == "__main__":
    main()
