"""
src/training/trainer.py

Trainer class — encapsulates the full train / validate loop.

Features:
  • tqdm progress bars
  • TensorBoard logging
  • Best-model checkpoint saving
  • Learning-rate scheduling (cosine / step / plateau)
  • Backbone unfreezing after cfg['model']['unfreeze_epoch'] epochs
  • Label-smoothing cross-entropy loss
"""

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.classifier import unfreeze_backbone
from .early_stopping import EarlyStopping


class Trainer:
    """
    Manages the training and validation loops.

    Args:
        model      : The nn.Module to train.
        cfg        : Parsed config.yaml dictionary.
        device     : torch.device ('cuda' or 'cpu').
        run_name   : Optional experiment name for TensorBoard.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        device: torch.device,
        run_name: str = "run",
    ):
        self.model  = model.to(device)
        self.cfg    = cfg
        self.device = device

        # ---- Loss ----
        label_smoothing = cfg["training"].get("label_smoothing", 0.1)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # ---- Optimizer (head params only at first) ----
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )

        # ---- Scheduler ----
        sched_name = cfg["training"].get("scheduler", "cosine")
        epochs = cfg["training"]["epochs"]
        if sched_name == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        elif sched_name == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=cfg["training"].get("step_size", 10),
                gamma=cfg["training"].get("gamma", 0.1),
            )
        elif sched_name == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")

        # ---- Early stopping ----
        self.early_stopping = EarlyStopping(
            patience=cfg["training"]["early_stopping_patience"],
            mode="min",
        )

        # ---- Paths ----
        self.best_model_path = Path(cfg["paths"]["best_model"])
        self.last_model_path = Path(cfg["paths"]["last_model"])
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

        # ---- TensorBoard ----
        log_dir = Path(cfg["paths"]["logs"]) / run_name
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # ---- State ----
        self.best_val_loss = float("inf")
        self.best_val_acc  = 0.0
        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

    # ------------------------------------------------------------------
    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one training epoch. Returns {'loss': ..., 'acc': ...}."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(loader, desc=f"  Epoch {epoch:02d} [train]", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Metrics
            total_loss += loss.item() * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / total
        accuracy = correct / total
        return {"loss": avg_loss, "acc": accuracy}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one validation epoch. Returns {'loss': ..., 'acc': ...}."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(loader, desc=f"  Epoch {epoch:02d} [val]  ", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss   = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return {"loss": avg_loss, "acc": accuracy}

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, val_loss: float, tag: str) -> None:
        path = self.best_model_path if tag == "best" else self.last_model_path
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "best_val_acc": self.best_val_acc,
                "cfg": self.cfg,
            },
            path,
        )

    # ------------------------------------------------------------------
    def _update_optimizer_backbone(self) -> None:
        """
        After unfreezing, re-initialise optimizer with two param groups:
          - backbone params: lower LR
          - head params:     original LR
        """
        backbone_lr = self.cfg["training"].get("backbone_lr", 1e-4)
        head_lr     = self.cfg["training"]["learning_rate"]

        head_name = "fc" if self.cfg["model"]["backbone"] == "resnet50" else "classifier"

        head_params     = [p for n, p in self.model.named_parameters()
                           if head_name in n and p.requires_grad]
        backbone_params = [p for n, p in self.model.named_parameters()
                           if head_name not in n and p.requires_grad]

        self.optimizer = AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params,     "lr": head_lr},
            ],
            weight_decay=self.cfg["training"]["weight_decay"],
        )

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> Dict[str, list]:
        """
        Full training loop.

        Returns:
            history dict with train/val loss and accuracy per epoch.
        """
        epochs         = self.cfg["training"]["epochs"]
        unfreeze_epoch = self.cfg["model"].get("unfreeze_epoch", 5)
        unfrozen       = False

        print(f"\n  Training for up to {epochs} epochs on {self.device}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # --- Backbone unfreeze ---
            if epoch == unfreeze_epoch and not unfrozen:
                print(f"\n  [Epoch {epoch}] Unfreezing backbone for fine-tuning ...")
                unfreeze_backbone(self.model, self.cfg)
                self._update_optimizer_backbone()
                unfrozen = True

            # --- Train + Validate ---
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics   = self.validate_epoch(val_loader, epoch)

            # --- Scheduler step ---
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["loss"])
            else:
                self.scheduler.step()

            # --- Logging ---
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:02d}/{epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"train_acc={train_metrics['acc']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_acc={val_metrics['acc']:.4f}  "
                f"({elapsed:.1f}s)"
            )

            self.writer.add_scalars(
                "Loss",
                {"train": train_metrics["loss"], "val": val_metrics["loss"]},
                epoch,
            )
            self.writer.add_scalars(
                "Accuracy",
                {"train": train_metrics["acc"], "val": val_metrics["acc"]},
                epoch,
            )
            cur_lr = self.optimizer.param_groups[-1]["lr"]
            self.writer.add_scalar("LR", cur_lr, epoch)

            # --- History ---
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["acc"])

            # --- Checkpoint ---
            self._save_checkpoint(epoch, val_metrics["loss"], tag="last")
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_val_acc  = val_metrics["acc"]
                self._save_checkpoint(epoch, val_metrics["loss"], tag="best")
                print(f"  ✓ Best model saved  (val_loss={self.best_val_loss:.4f})")

            # --- Early stopping ---
            if self.early_stopping(val_metrics["loss"]):
                print(f"\n  Early stopping triggered at epoch {epoch}.")
                break

        self.writer.close()
        print(f"\n  Training complete.  Best val_acc = {self.best_val_acc:.4f}")
        return self.history
