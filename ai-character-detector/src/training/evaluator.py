"""
src/training/evaluator.py

Evaluation pipeline.

  evaluate_model()  — runs inference on the test DataLoader and returns
                      a full metrics report + confusion matrix.

  plot_training_curves() — saves loss/accuracy plots from history dict.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

CLASS_NAMES = ["Real Human", "Cartoon / Anime", "AI Generated"]
CLASS_KEYS  = ["human", "cartoon", "ai_generated"]


# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_dir: str | Path = "logs",
) -> Dict:
    """
    Run full evaluation on a DataLoader.

    Returns a dict with:
      accuracy, precision, recall, f1,
      per_class (dict),
      confusion_matrix (np.ndarray),
      report (str)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Evaluating"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Scalar metrics ---
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES)))
    )

    # --- Print ---
    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (weighted)")
    print(f"  Recall    : {rec:.4f}  (weighted)")
    print(f"  F1 Score  : {f1:.4f}  (weighted)")
    print("\n" + report)

    # --- Confusion matrix plot ---
    _plot_confusion_matrix(cm, save_dir / "confusion_matrix.png")

    return {
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1":               f1,
        "confusion_matrix": cm,
        "report":           report,
    }


# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------

def plot_training_curves(history: Dict[str, list], save_dir: str | Path = "logs") -> None:
    """
    Save loss and accuracy curves from the trainer history dict.

    Args:
        history  : Dict with keys train_loss, val_loss, train_acc, val_acc.
        save_dir : Directory to save PNG files.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # --- Loss ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train loss", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val loss",   linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=150)
    plt.close()

    # --- Accuracy ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], label="Train accuracy", linewidth=2)
    ax.plot(epochs, history["val_acc"],   label="Val accuracy",   linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_curve.png", dpi=150)
    plt.close()

    print(f"  Training curves saved → {save_dir}/")
