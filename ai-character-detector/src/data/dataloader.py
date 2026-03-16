"""
src/data/dataloader.py

Factory that returns train, val, and test DataLoaders
configured from config.yaml values.
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from .dataset import CharacterDataset
from .transforms import get_train_transforms, get_val_transforms


def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader, test_loader).

    Args:
        cfg : Parsed config.yaml dictionary.

    Returns:
        Tuple of three DataLoaders.
    """
    proc = Path(cfg["paths"]["processed_data"])
    batch  = cfg["training"]["batch_size"]
    nw     = cfg["data"]["num_workers"]
    pin    = cfg["data"].get("pin_memory", True)

    train_ds = CharacterDataset(
        root=proc / "train",
        transform=get_train_transforms(cfg),
    )
    val_ds = CharacterDataset(
        root=proc / "val",
        transform=get_val_transforms(cfg),
    )
    test_ds = CharacterDataset(
        root=proc / "test",
        transform=get_val_transforms(cfg),
    )

    # Print split summary
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        print(f"  [{name}] {len(ds)} images  {ds.class_counts()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader
