"""
src/training/early_stopping.py

EarlyStopping utility.  Monitors a metric (default: validation loss)
and signals when training should stop after `patience` epochs without
improvement.
"""

import numpy as np


class EarlyStopping:
    """
    Stop training if monitored metric does not improve for `patience` epochs.

    Args:
        patience  : Number of epochs to wait after last improvement.
        min_delta : Minimum change in monitored value to qualify as improvement.
        mode      : 'min' (loss) or 'max' (accuracy / F1).
        verbose   : Print a message when patience counter increases.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.verbose   = verbose

        self.best_score: float | None = None
        self.counter   = 0
        self.stop      = False

    # ------------------------------------------------------------------
    def __call__(self, metric: float) -> bool:
        """
        Update state given the latest metric value.

        Returns:
            True  → caller should stop training.
            False → continue training.
        """
        score = -metric if self.mode == "min" else metric

        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(
                    f"  [early_stop] No improvement for {self.counter}/{self.patience} epochs."
                )
            if self.counter >= self.patience:
                self.stop = True
                return True
        else:
            self.best_score = score
            self.counter    = 0

        return False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.best_score = None
        self.counter    = 0
        self.stop       = False
