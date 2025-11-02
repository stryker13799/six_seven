from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Subset
from torchvision import datasets


@dataclass
class ImageSplits:
    train: Subset
    val: Subset
    test: Subset


def create_image_splits(
    root: Path | str,
    train_transform,
    eval_transform,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> ImageSplits:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"missing image root: {root}")
    full_train = datasets.ImageFolder(root=str(root), transform=train_transform)
    full_val = datasets.ImageFolder(root=str(root), transform=eval_transform)
    full_test = datasets.ImageFolder(root=str(root), transform=eval_transform)
    indices = np.arange(len(full_train))
    if len(indices) == 0:
        raise ValueError("image dataset is empty")
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    train_len = int(len(indices) * train_ratio)
    val_len = int(len(indices) * val_ratio)
    test_len = len(indices) - train_len - val_len
    if test_len <= 0:
        test_len = max(1, len(indices) - train_len - val_len)
    train_indices = indices[:train_len]
    val_indices = indices[train_len : train_len + val_len]
    test_indices = indices[train_len + val_len : train_len + val_len + test_len]
    return ImageSplits(
        train=Subset(full_train, train_indices.tolist()),
        val=Subset(full_val, val_indices.tolist()),
        test=Subset(full_test, test_indices.tolist()),
    )
