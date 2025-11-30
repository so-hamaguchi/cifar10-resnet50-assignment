import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights, resnet50
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

# クラス名の定義
CLASS_NAMES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

def set_seed(seed=42):
    """再現性のためにシードを固定する"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

def get_transforms():
    """データの前処理と拡張を定義する"""
    # 学習用：データ拡張あり
    # Resize((224, 224))と明示することで、意図しないアスペクト比の変更を防ぐ
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.RandomCrop(224, padding=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 検証用：リサイズと正規化のみ
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    return train_transform, val_transform

def get_dataloaders(batch_size, train_transform, val_transform, num_workers=2):
    """データローダーを作成する"""
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)

    # GPUが利用可能な場合は pin_memory=True にしてホストメモリ上のデータをページロックし、
    # GPUへの転送効率を上げる
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=use_pin_memory
    )

    return train_loader, val_loader

def build_model(device):
    """ResNet50モデルを構築し、最終層をCIFAR-10用に変更する"""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # CIFAR-10は10クラスなので出力数を変更
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """1エポック分の学習を実行する"""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    for imgs, labels in dataloader:
        # non_blocking=True でデータ転送待ちによるブロッキングを防ぐ
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = torch.argmax(output, dim=1)
        running_acc += torch.mean(pred.eq(labels).float()).item()

    avg_loss = running_loss / len(dataloader)
    avg_acc = running_acc / len(dataloader)
    return avg_loss, avg_acc

def validate(model, dataloader, criterion, device):
    """検証を実行する"""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            # non_blocking=True
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            output = model(imgs)
            loss = criterion(output, labels)
            
            running_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            running_acc += torch.mean(pred.eq(labels).float()).item()
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    avg_acc = running_acc / len(dataloader)
    
    # 返り値の順序を修正: ラベル(正解), 予測 の順にする
    return avg_loss, avg_acc, all_labels, all_preds

def save_plots(train_losses, val_losses, train_accs, val_accs, all_labels, all_preds, output_dir="images"):
    """学習曲線と混同行列を保存する"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Loss & Accuracy Curve
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_accs, label='Train Acc', marker='o')
    plt.plot(epochs_range, val_accs, label='Val Acc', marker='o')
    plt.title('Accuracy: Train vs Val')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Val Loss', marker='o')
    plt.title('Loss: Train vs Val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_accuracy_curve.png"))
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    print(f"Plots saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Training with ResNet50")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    args = parser.parse_args()

    # シード固定
    set_seed(args.seed)
    
    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # WandB初期化
    if args.use_wandb:
        wandb.init(
            project="cifar10-resnet-production",
            config=vars(args),
            name=f"resnet50_e{args.epochs}_b{args.batch_size}"
        )

    # データ準備
    train_transform, val_transform = get_transforms()
    train_loader, val_loader = get_dataloaders(args.batch_size, train_transform, val_transform)

    # モデル・損失関数・オプティマイザ・スケジューラ
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 学習ループ
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0

    print("Start Training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        
        # 修正: validateの返り値に合わせて受け取る変数の順序を変更
        val_loss, val_acc, val_labels, val_preds = validate(model, val_loader, criterion, device)
        
        # 指標計算 (Precision, Recall, F1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='macro', zero_division=0
        )
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"F1: {f1:.4f}")

        # W&Bログ
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
                "lr": scheduler.get_last_lr()[0]
            })

        # ベストモデルの保存
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/best_model.pth")
            print(f"  -> Best model saved with accuracy: {best_acc:.4f}")

    # 最終結果のグラフ保存
    # save_plotsは (all_labels, all_preds) の順で受け取るのでそのまま渡す
    save_plots(train_losses, val_losses, train_accs, val_accs, val_labels, val_preds)

    if args.use_wandb:
        wandb.finish()
    
    print("Training Finished.")

if __name__ == "__main__":
    main()
