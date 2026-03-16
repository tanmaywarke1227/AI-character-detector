"""
download_dataset.py

Downloads and prepares the dataset for ai-character-detector.

Sources:
  - CelebA (human faces)          → data/raw/human/
  - Danbooru via HuggingFace      → data/raw/cartoon/
  - AI-generated images (manual)  → data/raw/ai_generated/

After downloading, splits into:
  data/processed/train/  (80%)
  data/processed/val/    (10%)
  data/processed/test/   (10%)

Usage:
  python download_dataset.py --source all
  python download_dataset.py --source human
  python download_dataset.py --source cartoon
  python download_dataset.py --split-only
"""

import os
import shutil
import random
import argparse
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download_file(url: str, dest: Path, desc: str = "") -> None:
    """Stream-download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=desc, total=total, unit="B", unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    zip_path.unlink()  # remove zip after extraction


# ---------------------------------------------------------------------------
# Downloaders
# ---------------------------------------------------------------------------

def download_celeba(raw_dir: Path, max_images: int = 30000) -> None:
    """
    Download CelebA dataset (human faces).

    CelebA is hosted on Google Drive. We use gdown for convenience.
    Only the img_align_celeba.zip is needed (aligned face crops).

    If gdown is unavailable or the download fails, this function prints
    manual instructions.
    """
    human_dir = raw_dir / "human"
    human_dir.mkdir(parents=True, exist_ok=True)

    existing = list(human_dir.glob("*.jpg")) + list(human_dir.glob("*.png"))
    if len(existing) >= max_images:
        print(f"  [human] {len(existing)} images already present — skipping download.")
        return

    try:
        import gdown  # type: ignore
    except ImportError:
        print("  [human] gdown not installed. Run: pip install gdown")
        print("  Then re-run this script, or manually place images in data/raw/human/")
        return

    # CelebA aligned images (~1.3 GB)
    gdrive_id = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    zip_dest = raw_dir / "celeba.zip"

    print("  Downloading CelebA (aligned faces) from Google Drive ...")
    try:
        gdown.download(id=gdrive_id, output=str(zip_dest), quiet=False)
        extract_zip(zip_dest, raw_dir / "_celeba_tmp")

        # Move images into human/
        src = raw_dir / "_celeba_tmp" / "img_align_celeba"
        images = sorted(src.glob("*.jpg"))[:max_images]
        print(f"  Copying {len(images)} images to {human_dir} ...")
        for img in tqdm(images, desc="  human"):
            shutil.copy(img, human_dir / img.name)
        shutil.rmtree(raw_dir / "_celeba_tmp", ignore_errors=True)
        print(f"  [human] Done — {len(images)} images saved.")
    except Exception as e:
        print(f"  [human] Download failed: {e}")
        print("  Manually place JPG/PNG images in data/raw/human/ and re-run with --split-only")


def download_danbooru(raw_dir: Path, max_images: int = 30000) -> None:
    """
    Download anime/cartoon images via HuggingFace datasets.

    Uses the 'animefacedataset' or 'danbooru-aesthetic' subset.
    Requires: pip install datasets
    """
    cartoon_dir = raw_dir / "cartoon"
    cartoon_dir.mkdir(parents=True, exist_ok=True)

    existing = list(cartoon_dir.glob("*.jpg")) + list(cartoon_dir.glob("*.png"))
    if len(existing) >= max_images:
        print(f"  [cartoon] {len(existing)} images already present — skipping download.")
        return

    try:
        from datasets import load_dataset  # type: ignore
        from PIL import Image as PILImage
    except ImportError:
        print("  [cartoon] Install HuggingFace datasets: pip install datasets")
        print("  Or manually place images in data/raw/cartoon/")
        return

    print("  Downloading anime face dataset from HuggingFace ...")
    try:
        ds = load_dataset("animefacedataset", split="train", streaming=True)
        count = 0
        for sample in tqdm(ds, total=max_images, desc="  cartoon"):
            if count >= max_images:
                break
            img = sample.get("image") or sample.get("img")
            if img is None:
                continue
            if not isinstance(img, PILImage.Image):
                continue
            img.save(cartoon_dir / f"cartoon_{count:06d}.jpg")
            count += 1
        print(f"  [cartoon] Done — {count} images saved.")
    except Exception as e:
        print(f"  [cartoon] Download failed: {e}")
        print("  Manually place images in data/raw/cartoon/ and re-run with --split-only")


def prepare_ai_generated(raw_dir: Path) -> None:
    """
    AI-generated images must be added manually (or generated via diffusion models).

    This function creates the folder and prints instructions.
    Popular free sources:
      - https://thispersondoesnotexist.com  (single images, scrape carefully)
      - https://huggingface.co/datasets/papluca/laion-face (mixed, filter AI)
      - Generate via local Stable Diffusion / SDXL
    """
    ai_dir = raw_dir / "ai_generated"
    ai_dir.mkdir(parents=True, exist_ok=True)

    existing = list(ai_dir.glob("*.jpg")) + list(ai_dir.glob("*.png"))
    if existing:
        print(f"  [ai_generated] {len(existing)} images already present.")
        return

    print("  [ai_generated] No images found.")
    print("  Please add AI-generated face images to:  data/raw/ai_generated/")
    print("  Recommended sources:")
    print("    • Generate with Stable Diffusion locally")
    print("    • https://thispersondoesnotexist.com")
    print("    • HuggingFace: papluca/laion-face (filter for AI-generated subset)")
    print("  Target: ~30,000 images for balanced training.")


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

def split_dataset(cfg: dict) -> None:
    """
    Splits raw images into processed/train, processed/val, processed/test.

    Structure after splitting:
      data/processed/
        train/human/   train/cartoon/   train/ai_generated/
        val/human/     val/cartoon/     val/ai_generated/
        test/human/    test/cartoon/    test/ai_generated/
    """
    raw_dir = Path(cfg["paths"]["raw_data"])
    proc_dir = Path(cfg["paths"]["processed_data"])
    classes = cfg["classes"]

    train_ratio = cfg["data"]["train_split"]
    val_ratio = cfg["data"]["val_split"]
    # test_ratio is remainder

    total_moved = 0

    for cls in classes:
        src_dir = raw_dir / cls
        if not src_dir.exists():
            print(f"  [split] Warning: {src_dir} not found — skipping.")
            continue

        images = (
            list(src_dir.glob("*.jpg"))
            + list(src_dir.glob("*.jpeg"))
            + list(src_dir.glob("*.png"))
            + list(src_dir.glob("*.webp"))
        )

        if not images:
            print(f"  [split] No images in {src_dir} — skipping.")
            continue

        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        print(f"  [split] {cls}: {n} images → "
              f"train={n_train}, val={n_val}, test={n - n_train - n_val}")

        for split_name, split_images in splits.items():
            dest_dir = proc_dir / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_path in tqdm(split_images, desc=f"    {split_name}/{cls}", leave=False):
                shutil.copy(img_path, dest_dir / img_path.name)
            total_moved += len(split_images)

    print(f"\n  [split] Complete — {total_moved} images organised in {proc_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download and prepare dataset")
    parser.add_argument(
        "--source",
        choices=["all", "human", "cartoon", "ai"],
        default="all",
        help="Which dataset source to download",
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Skip download, only run the train/val/test split",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=30000,
        help="Max images per class to download (default: 30000)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = load_config(args.config)
    raw_dir = Path(cfg["paths"]["raw_data"])

    if not args.split_only:
        print("\n=== Downloading datasets ===")
        if args.source in ("all", "human"):
            print("\n[1/3] CelebA (human faces)")
            download_celeba(raw_dir, max_images=args.max_images)

        if args.source in ("all", "cartoon"):
            print("\n[2/3] Anime/Cartoon faces")
            download_danbooru(raw_dir, max_images=args.max_images)

        if args.source in ("all", "ai"):
            print("\n[3/3] AI-generated images")
            prepare_ai_generated(raw_dir)

    print("\n=== Splitting dataset ===")
    split_dataset(cfg)

    print("\n=== Dataset preparation complete ===")
    print("Next step: python train.py")


if __name__ == "__main__":
    main()
