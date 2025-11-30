# CIFAR-10 Classification with ResNet50

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![WandB](https://img.shields.io/badge/WandB-Log-orange?logo=weightsandbiases&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

このリポジトリは、データセット**CIFAR-10**に対して**ResNet50**を用いた画像分類モデルを実装し、各種改善施策(データ拡張・正則化・転移学習・学習率調整)を検証するためのコード一式をまとめたものです。

## 実験結果

転移学習（全層ファインチューニング）と学習率スケジューリングなどの導入により、検証データの精度をかなり高めることができました。

| モデル設定 | Epochs | 手法 | Best Val Acc |
| :--- | :---: | :--- | :---: |
| Baseline (scratch) | 15 | スクラッチ学習・正規化のみ | 67.2% |
| **Final Model** | **15** | **データ拡張・正則化 + 全層FT + CosineScheduler** | **97.7%**|

### 学習曲線の推移
![Learning Curve](images/loss_accuracy_curve.png)

*図1: 最終モデルにおけるAccuracyとLossの推移*

### 混同行列 (Confusion Matrix)
![Confusion Matrix](images/confusion_matrix.png)

*図2: クラスごとの予測精度。CatとDogの間でわずかな誤分類が見られるものの、全体として極めて高い分類能力を示している。*



## 環境構築
- python 3.10以降

- pytorch/torchvision

- WandB (ログ管理用)

必要なライブラリは`requirements.txt`をご参照ください。


### ディレクトリ構成

```text
.
├── train.py           # 学習用スクリプト
├── predict.py         # 推論用スクリプト
├── requirements.txt   # 依存ライブラリ
├── README.md          # ドキュメント
├── images/            # 実験結果の画像
│   ├── loss_accuracy_curve_base.png # 97.7%モデルの学習曲線（配布用）
│   ├── confusion_matrix_base.png    # 97.7%モデルの混同行列（配布用）
│   ├── loss_accuracy_curve.png      # train.py実行時に生成される
│   └── confusion_matrix.png         # train.py実行時に生成される
├── notebooks/         # 実験用ノートブック（全９ファイル）
├── weights/           # 学習済みモデル
│   ├── best_model_base.pth # 97.7%達成のモデル（配布用）
│   └── best_model.pth      # train.py実行時に生成される
└── slides/
    └── report.pdf     # レポート用スライド
```

### インストール手順

```
# リポジトリのクローン
git clone https://github.com/so-hamaguchi/cifar10-resnet50-assignment.git
cd cifar10-resnet50-assignment

# 必要なライブラリのインストール
pip install -r requirements.txt
```

## 学習実行手順
- デフォルト実行

```
python train.py
```

- W&Bにログを記録したい場合（任意）

```
# W&Bにログイン
wandb login

python train.py --use_wandb
```

- ハイパーパラメータの変更

```
#例
python train.py --epochs 12 --batch_size 64 --seed 42
```
Out of Memoryになってしまう場合は、batch_sizeを32などに下げてお試しください。

### 出力
`python train.py` の実行後、以下が自動生成されます。

- 学習済み重み：`weights/best_model.pth`
- 学習のログ：`images/loss_accuracy_curve.png`
- 最終epochでの混同行列：`images/confusion_matrix.png`

## 推論実行手順

学習済みモデルを使用して、画像のクラス分類を行います。
- 猫の画像

```
# 実行例
python predict.py --image samples/cat.jpg --weights weights/best_model_base.pth
```

- 船の画像

```
# 実行例
python predict.py --image samples/ship.jpg --weights weights/best_model_base.pth
```



ご自身で train.py を実行して生成されたモデルを使用する場合、学習済みモデルは `best_model.pth` と保存されます。
以下のように、変更をお願いします。

```
# 実行例
python predict.py --image samples/cat.jpg --weights weights/best_model.pth
```

## 再現性
- 乱数シードはデフォルトで `42` に固定しています。（`--seed` で変更可能）
- 再現性重視のため、cuDNNの設定を固定しています。

## 改善施策
- Data Augmentation: flip/crop/rotate
- Regularization: weight decay
- Transfer Learning: ImageNet pretrained ResNet50, full fine-tuning
- LR Scheduler: CosineAnnealingLR

/Notebookのフォルダの中にそれぞれの改善施策の実験ノートブックが格納されています。

## W&B 実験ログ
- Project: https://wandb.ai/so-hamaguchi-student/cifar10-resnet?nw=nwusersohamaguchi
- Best run: https://wandb.ai/so-hamaguchi-student/cifar10-resnet/runs/oxsv8ssg?nw=nwusersohamaguchi

## License
MIT License
