from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast # pyright: ignore[reportPrivateImportUsage]
from torch.utils.data import DataLoader
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gesture67 import (
    AverageMeter,
    accuracy_topk,
    build_mobilenet,
    create_image_splits,
    set_seed,
)


class AlbumentationsWrapper:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, img):
        return self.aug(image=np.array(img))["image"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV3 on the 67 dataset")
    parser.add_argument("--image-root", type=Path, default=Path("data/images"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/mobilenet_v3.pth"))
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/mobilenet_v3_metrics.json"))
    return parser.parse_args()


def build_transforms() -> tuple:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    train_tf = A.Compose([
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return train_tf, eval_tf


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mobilenet()
    train_tf, eval_tf = build_transforms()

    splits = create_image_splits(
        args.image_root,
        AlbumentationsWrapper(train_tf),
        eval_tf,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=args.seed,
    )
    train_loader = DataLoader(
        splits.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        splits.val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        splits.test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = None
    effective_epochs = max(1, args.epochs - args.warmup_epochs)
    if effective_epochs > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=effective_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val = 0.0
    history = []
    print(f"\nStarting training on {device}")
    print(f"Train: {len(splits.train)} | Val: {len(splits.val)} | Test: {len(splits.test)}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}\n")
    print(f"Device: {device}")
    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            warm_lr = args.lr * float(epoch + 1) / max(1, args.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warm_lr
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            top1 = accuracy_topk(logits, targets, topk=(1,))[0]
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(top1.item(), images.size(0))
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_meter.avg,
                "train_acc": acc_meter.avg,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        curr_lr = optimizer.param_groups[0]["lr"]
        marker = "*" if val_acc > best_val else " "
        print(f"[{epoch+1:>3}/{args.epochs}] "
              f"loss: {loss_meter.avg:.4f} | acc: {acc_meter.avg:5.2f}% | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:5.2f}% | "
              f"lr: {curr_lr:.2e} {marker}")
        if val_acc > best_val:
            best_val = val_acc
            save_checkpoint(model, args.checkpoint)
        if epoch >= args.warmup_epochs and scheduler is not None:
            scheduler.step()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)
    test_acc, test_loss = evaluate(model, test_loader, criterion, device)
    print(f"\nTest results: loss={test_loss:.4f} | acc={test_acc:.2f}%")
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved metrics to {args.metrics_path}")
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_path.open("w") as fh:
        json.dump(
            {
                "best_val_acc": best_val,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "history": history,
            },
            fh,
            indent=2,
        )


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            top1 = accuracy_topk(logits, targets, topk=(1,))[0]
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(top1.item(), images.size(0))
    return acc_meter.avg, loss_meter.avg


def save_checkpoint(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    main()
