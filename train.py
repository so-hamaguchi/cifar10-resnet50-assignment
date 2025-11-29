"""
ResNet50を用いたCIFAR-10画像分類モデルの学習スクリプト。

Features:
- Transfer Learning: ResNet50 (ImageNet pretrained)
- GPU-accelerated augmentation using torchvision.transforms.v2
- Cosine Annealing LR scheduler
- Optional experiment tracking with Weights & Biases (W&B)
  - Enable by passing: --use_wandb
  - Otherwise runs with W&B disabled (no login required)

Usage:
    python train.py
    python train.py --use_wandb
"""

from __future__ import annotations

import argparse
import os
import random
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import v2

import matplotlib.pyplot as plt
import seaborn as sns
import wandb


CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass(frozen=True)
class Config:
    """Training configuration."""

    learning_rate: float = 1e-4
    batch_size: int = 256
    epochs: int = 12
    weight_decay: float = 1e-4
    optimizer: str = "Adam"
    architecture: str = "ResNet50"
    dataset: str = "CIFAR-10"
    augmentation: str = "v2_gpu_flip_crop_rotate"
    note: str = "全層FT + CosineAnnealingLR + GPU Augmentation"
    seed: int = 42
    num_workers: int = 2
    out_dir: str = "outputs"
    save_name: str = "resnet50_cifar10.pth"  # saved under out_dir/checkpoints/


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="CIFAR-10 training with ResNet50")
    p.add_argument("--use_wandb", action="store_true", help="Enable W&B logging (no login required if API key is set).")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs in config.")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch_size in config.")
    p.add_argument("--seed", type=int, default=None, help="Override seed in config.")
    p.add_argument("--log_images", action="store_true", help="Log misclassified images (default: last epoch only).")
    return p.parse_args()


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility.

    Notes:
        cudnn.deterministic=True improves reproducibility but can slow training.
        cudnn.benchmark=False avoids non-deterministic algorithm selection.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gpu_transforms(device: torch.device) -> tuple[v2.Compose, v2.Compose]:
    """Build GPU-side transforms (v2 API).

    - Resize to 224x224 to match ImageNet-pretrained ResNet50 input resolution.
    - Normalize with ImageNet mean/std.
    """
    # Train: augmentation
    train_t = v2.Compose(
        [
            v2.Resize((224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=5),
            v2.RandomCrop((224, 224), padding=3),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ).to(device)

    # Val: deterministic
    val_t = v2.Compose(
        [
            v2.Resize((224, 224), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ).to(device)

    return train_t, val_t


def get_dataloaders(batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    """Create DataLoaders.
    CPU-side does only ToTensor; heavy transforms run on GPU via v2.
    """
    base_transform = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=base_transform)
    val_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=base_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(device: torch.device) -> nn.Module:
    """Build ResNet50 for CIFAR-10."""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    gpu_transform: v2.Compose,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch and return (loss, accuracy)."""
    model.train()
    loss_sum = 0.0
    acc_sum = 0.0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        imgs = gpu_transform(imgs)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item())
        preds = torch.argmax(logits, dim=1)
        acc_sum += float((preds == labels).float().mean().item())

    return loss_sum / len(loader), acc_sum / len(loader)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    gpu_transform: v2.Compose,
    device: torch.device,
    collect_misclassified: bool = False,
    max_misclassified: int = 32,
) -> dict:
    """Validate and return metrics.

    Returns:
        dict with keys: loss, acc, precision, recall, f1, cm, misclassified
        misclassified is List[Tuple[np.ndarray, str]] (image, caption) if collected.
    """
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    misclassified: List[Tuple[np.ndarray, str]] = []

    # inverse normalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            imgs_t = gpu_transform(imgs)
            logits = model(imgs_t)
            loss = criterion(logits, labels)

            loss_sum += float(loss.item())
            preds = torch.argmax(logits, dim=1)
            acc_sum += float((preds == labels).float().mean().item())

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

            if collect_misclassified and len(misclassified) < max_misclassified:
                mistakes = preds != labels
                idxs = mistakes.nonzero(as_tuple=True)[0]
                for idx in idxs:
                    if len(misclassified) >= max_misclassified:
                        break
                    # de-normalize
                    img = imgs_t[idx : idx + 1] * std + mean  # [1,3,H,W]
                    img_np = img.squeeze(0).detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    cap = f"Pred: {CLASSES[preds[idx].item()]}, True: {CLASSES[labels[idx].item()]}"
                    misclassified.append((img_np, cap))

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "loss": loss_sum / len(loader),
        "acc": acc_sum / len(loader),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "cm": cm,
        "misclassified": misclassified,
    }


def plot_confusion_matrix(cm: np.ndarray, title: str) -> plt.Figure:
    """Create confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def ensure_dirs(cfg: Config) -> dict:
    """Create output directories and return paths."""
    out_dir = Path(cfg.out_dir)
    fig_dir = out_dir / "figures"
    ckpt_dir = out_dir / "checkpoints"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return {"out": out_dir, "fig": fig_dir, "ckpt": ckpt_dir}


def main() -> None:
    args = parse_args()
    cfg = Config(
        epochs=args.epochs if args.epochs is not None else Config.epochs,
        batch_size=args.batch_size if args.batch_size is not None else Config.batch_size,
        seed=args.seed if args.seed is not None else Config.seed,
    )

    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    paths = ensure_dirs(cfg)

    # W&B: enabled only when --use_wandb, otherwise disabled (no login required)
    wandb_mode = "online" if args.use_wandb else "disabled"
    run = wandb.init(
        project="cifar10-resnet",
        name="resnet50_v2_gpu_aug",
        config=cfg.__dict__,
        mode=wandb_mode,
    )

    gpu_train_t, gpu_val_t = get_gpu_transforms(device)
    train_loader, val_loader = get_dataloaders(cfg.batch_size, cfg.num_workers)

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    log_path = paths["out"] / "log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"])

        print("Start training...")

        for epoch in range(cfg.epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, gpu_train_t, device
            )
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

            collect_imgs = args.log_images or (epoch == cfg.epochs - 1)
            val_metrics = validate(
                model,
                val_loader,
                criterion,
                gpu_val_t,
                device,
                collect_misclassified=collect_imgs,
                max_misclassified=32,
            )

            # CSV
            writer.writerow([
                epoch + 1,
                lr,
                train_loss,
                train_acc,
                val_metrics["loss"],
                val_metrics["acc"],
                val_metrics["f1"],
            ])
            f.flush()

            # Console
            print(
                f"Epoch [{epoch+1}/{cfg.epochs}] "
                f"LR: {lr:.2e} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} "
                f"F1: {val_metrics['f1']:.4f}"
            )

            # Confusion matrix: W&Bなら送る / なければ保存
            cm_fig = plot_confusion_matrix(val_metrics["cm"], title=f"Confusion Matrix (Epoch {epoch+1})")
            if args.use_wandb:
                cm_img = wandb.Image(cm_fig)
            else:
                cm_path = paths["fig"] / f"confusion_matrix_epoch_{epoch+1:03d}.png"
                cm_fig.savefig(cm_path)
                cm_img = None
            plt.close(cm_fig)

            # W&B Log
            if args.use_wandb:
                misclassified_wandb = [
                    wandb.Image(img, caption=cap) for img, cap in val_metrics["misclassified"]
                ]
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "lr": lr,
                        "train/loss": train_loss,
                        "train/acc": train_acc,
                        "val/loss": val_metrics["loss"],
                        "val/acc": val_metrics["acc"],
                        "val/precision": val_metrics["precision"],
                        "val/recall": val_metrics["recall"],
                        "val/f1": val_metrics["f1"],
                        "val/confusion_matrix": cm_img,
                        "val/misclassified": misclassified_wandb,
                    }
                )

        
    # Save model
    save_path = paths["ckpt"] / cfg.save_name
    torch.save(model.state_dict(), save_path)
    print(f"Training finished. Model saved to: {save_path}")

    if args.use_wandb:
        wandb.save(str(save_path))
        wandb.finish()


if __name__ == "__main__":
    main()
