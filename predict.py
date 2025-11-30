import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
import os
import sys

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


def get_args():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Image Classifier using ResNet50"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the input image file"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="experiment_adjustment.pth",
        help="Path to the trained model weights (.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference",
    )
    return parser.parse_args()


def load_model(weights_path, device):
    """モデルを構築し、学習済み重みをロードする"""
    print("> Building ResNet50 model...")

    # 1. モデルの枠組みを作成
    model = models.resnet50(weights=None)

    # 2. 最終層をCIFAR-10用（10クラス）に書き換え
    model.fc = nn.Linear(model.fc.in_features, 10)

    # 3. 重みファイルのロード
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at '{weights_path}'")
        sys.exit(1)

    print(f"[info] Loading weights from '{weights_path}'...")
    try:
        # map_locationを使うことで、GPUで学習したモデルをCPUしか無い環境でもロード可能に
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        sys.exit(1)

    model.to(device)
    model.eval()  # 評価モードにする（DropoutやBatch Normを固定）
    return model


def preprocess_image(image_path):
    """画像を読み込み、モデルに入力できる形式に変換する"""
    # 学習時の 'val_transform' と同じ処理を行います
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # ResNet50の入力サイズに合わせる
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        sys.exit(1)

    try:
        image = Image.open(image_path).convert("RGB")  # pngなどのアルファチャンネル対策
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(
            0
        )  # バッチ次元を追加 [3, 224, 224] -> [1, 3, 224, 224]
        return input_batch
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)


def main():
    args = get_args()

    # デバイスの決定
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] Using device: {device}")

    # モデルと画像の準備
    model = load_model(args.weights, device)
    input_batch = preprocess_image(args.image).to(device)

    # 推論実行
    print(">>> Predicting...")
    with torch.no_grad():  # 勾配計算を無効化してメモリ節約・高速化
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # 上位3つの予測を表示
    top_prob, top_catid = torch.topk(probabilities, 3)

    print("\n" + "=" * 30)
    print(f"[Image]  Input: {args.image}")
    print("=" * 30)
    print(f"Top 1: {CLASSES[top_catid[0]]:<10} ({top_prob[0].item() * 100:.2f}%)")
    print("-" * 30)

    # 2位と3位も表示（デバッグや確認に便利）
    for i in range(1, 3):
        print(
            f"   Top {i + 1}: {CLASSES[top_catid[i]]:<10} ({top_prob[i].item() * 100:.2f}%)"
        )
    print("=" * 30 + "\n")


if __name__ == "__main__":
    main()
