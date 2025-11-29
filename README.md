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
| **Final Model** | **12** | **データ拡張・正則化 + 全層FT + CosineScheduler** | **97.7%**|

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
├── images/            # 実験結果の画像（グラフなど）
│   ├── loss_accuracy_curve.png
│   └── confusion_matrix.png
├── notebooks/         # 実験用ノートブック（全９ファイル）
├── weights/           # 学習済みモデル
│   └── best_model.pth
└── slides/
    └── report.pdf     # レポート用スライド（後で追加）
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

# --log_imagesとすると誤分類画像が毎エポック記録されます。必要な場合はご使用ください。
```

- ハイパーパラメータの変更

```
#例
python train.py --epochs 12 --batch_size 64 --seed 42
```
Out of Memoryになってしまう場合は、batch_sizeを32などに下げてお試しください。

### 出力
`python train.py` の実行後、以下が自動生成されます。

- 学習済み重み：`outputs/checkpoints/resnet50_cifar10.pth`
- 学習のログ：`outputs/log.csv`
- 図：`outputs/figures/`（混同行列など：W&B無効時）


## 推論実行手順

学習済みモデルを使用して、画像のクラス分類を行います。
リポジトリに含まれている精度97.7%のモデルを使用する場合は、下記のように実行してください

```
# 実行例
python predict.py --image samples/cat.jpg --weights weights/best_model.pth
```
さらに、--deviceでcpu/cudaが選べます。デフォルトはautoです。



ご自身で train.py を実行して生成されたモデルを使用する場合、学習済みモデルは `outputs/checkpoints/` に保存されます。

そのため、そのモデルを推論で使用される場合は以下のようにパスの変更をお願いします。

```
# 実行例
python predict.py --image samples/cat.jpg --weights outputs/checkpoints/resnet50_cifar10.pth
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

## License
MIT License
