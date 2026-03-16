"""
train.py

Main training entry point for the ai-character-detector.

Usage:
    # Train with defaults from config.yaml
    python train.py

    # Override backbone and batch size
    python train.py --backbone efficientnet_b0 --batch-size 64

    # Resume from a checkpoint
    python train.py --resume saved_models/last_model.pth

    # Dry run (1 batch per epoch, 2 epochs)
    python train.py --dry-run

TensorBoard:
    tensorboard --logdir logs/
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Overwrite config values with any CLI arguments that were provided."""
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    return cfg


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [device] GPU: {name}  ({mem:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  [device] Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("  [device] CPU — training will be slow")
    return device


def print_banner(cfg: dict) -> None:
    backbone = cfg["model"]["backbone"]
    epochs   = cfg["training"]["epochs"]
    bs       = cfg["training"]["batch_size"]
    lr       = cfg["training"]["learning_rate"]
    print("\n" + "=" * 60)
    print("  AI CHARACTER DETECTOR — TRAINING")
    print("=" * 60)
    print(f"  Backbone  : {backbone}")
    print(f"  Epochs    : {epochs}")
    print(f"  Batch size: {bs}")
    print(f"  LR        : {lr}")
    print(f"  Scheduler : {cfg['training'].get('scheduler', 'cosine')}")
    print(f"  Smoothing : {cfg['training'].get('label_smoothing', 0.1)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Dry-run dataloader wrapper (limits batches for quick smoke-tests)
# ---------------------------------------------------------------------------

class DryRunLoader:
    """Wraps a DataLoader to yield only `n_batches` batches per epoch."""

    def __init__(self, loader, n_batches: int = 2):
        self.loader   = loader
        self.n_batches = n_batches
        self.dataset  = loader.dataset

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.n_batches:
                break
            yield batch

    def __len__(self):
        return min(self.n_batches, len(self.loader))


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------

def maybe_resume(model, optimizer, resume_path: str | None, device: torch.device):
    """
    Load a checkpoint if --resume was specified.
    Returns the epoch to start from (0-indexed), or 0 if no resume.
    """
    if not resume_path:
        return 0

    path = Path(resume_path)
    if not path.exists():
        print(f"  [resume] Checkpoint not found: {path} — starting fresh.")
        return 0

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    print(f"  [resume] Resumed from epoch {start_epoch} ({path})")
    return start_epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the AI Character Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",     default="config.yaml", help="Path to config.yaml")
    p.add_argument("--backbone",   default=None, choices=["resnet50", "efficientnet_b0"])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--lr",         type=float, default=None, help="Learning rate")
    p.add_argument("--resume",     default=None, help="Path to checkpoint to resume from")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--dry-run",    action="store_true",
                   help="Run 2 batches per epoch for 2 epochs (smoke test)")
    p.add_argument("--eval-only",  action="store_true",
                   help="Skip training; only evaluate the saved best model on the test set")
    p.add_argument("--no-gradcam", action="store_true",
                   help="Skip Grad-CAM generation after training")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Setup ----
    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    if args.dry_run:
        cfg["training"]["epochs"] = 2
        print("  [dry-run] Only 2 batches per epoch, 2 epochs.")

    print_banner(cfg)
    device = get_device()

    # ---- Imports (here to allow --help without heavy deps) ----
    from src.data.dataloader import get_dataloaders
    from src.models.classifier import build_model, get_model_info
    from src.training.trainer import Trainer
    from src.training.evaluator import evaluate_model, plot_training_curves

    # ---- Data ----
    print("\n  Loading datasets ...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    if args.dry_run:
        train_loader = DryRunLoader(train_loader, n_batches=2)
        val_loader   = DryRunLoader(val_loader,   n_batches=2)
        test_loader  = DryRunLoader(test_loader,  n_batches=2)

    # ---- Model ----
    print("\n  Building model ...")
    model = build_model(cfg)
    info  = get_model_info(model)
    print(f"  Total params    : {info['total_params']:,}")
    print(f"  Trainable params: {info['trainable_params']:,}")

    # ---- Eval-only mode ----
    if args.eval_only:
        ckpt_path = Path(cfg["paths"]["best_model"])
        if not ckpt_path.exists():
            print(f"  ERROR: No model found at {ckpt_path}. Train first.")
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        print(f"\n  Evaluating {ckpt_path} on test set ...")
        evaluate_model(model, test_loader, device, save_dir=cfg["paths"]["logs"])
        return

    # ---- Run name for TensorBoard ----
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    backbone = cfg["model"]["backbone"]
    run_name = f"{backbone}_{ts}"

    # ---- Trainer ----
    trainer = Trainer(model=model, cfg=cfg, device=device, run_name=run_name)

    # ---- Optional resume ----
    maybe_resume(model, trainer.optimizer, args.resume, device)

    # ---- Train ----
    print(f"\n  Run name: {run_name}")
    print(f"  TensorBoard: tensorboard --logdir {cfg['paths']['logs']}/\n")

    t_start  = time.time()
    history  = trainer.fit(train_loader, val_loader)
    t_total  = time.time() - t_start

    print(f"\n  Total training time: {t_total/60:.1f} min")

    # ---- Training curves ----
    log_dir = Path(cfg["paths"]["logs"]) / run_name
    plot_training_curves(history, save_dir=log_dir)

    # ---- Final test evaluation ----
    print("\n  Running final evaluation on test set ...")
    best_ckpt = torch.load(cfg["paths"]["best_model"], map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    metrics = evaluate_model(model, test_loader, device, save_dir=log_dir)

    # ---- Grad-CAM sample (first test image) ----
    if not args.no_gradcam:
        _run_sample_gradcam(model, test_loader, cfg, device, log_dir)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best val accuracy : {trainer.best_val_acc:.4f}")
    print(f"  Test accuracy     : {metrics['accuracy']:.4f}")
    print(f"  Test F1 (weighted): {metrics['f1']:.4f}")
    print(f"  Best model saved  : {cfg['paths']['best_model']}")
    print(f"  Logs              : {log_dir}")
    print("=" * 60)
    print("\n  Next step: python predict.py --image path/to/image.jpg")
    print("  Or start the API: uvicorn src.api.server:app --reload\n")


# ---------------------------------------------------------------------------
# Grad-CAM sample helper
# ---------------------------------------------------------------------------

def _run_sample_gradcam(model, loader, cfg, device, save_dir: Path) -> None:
    """Generate a Grad-CAM overlay for the first image in the loader."""
    from src.data.transforms import denormalize
    from src.utils.gradcam import generate_gradcam, save_gradcam
    import torch.nn.functional as F

    try:
        images, labels = next(iter(loader))
        img_tensor = images[0:1].to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            pred   = int(logits.argmax(dim=1).item())

        original_rgb = denormalize(img_tensor)
        overlay = generate_gradcam(
            model=model,
            image_tensor=img_tensor,
            target_class=pred,
            cfg=cfg,
            original_image=original_rgb,
        )
        if overlay is not None:
            out_path = str(save_dir / "sample_gradcam.png")
            save_gradcam(overlay, out_path)
    except Exception as e:
        print(f"  [gradcam] Could not generate sample: {e}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
