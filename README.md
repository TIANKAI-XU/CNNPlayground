# CNNPlayground

一个基于 PyTorch 的 CNN 学习与实验仓库，用于复现和对比经典卷积神经网络，并扩展到简单的图像二分类任务。

当前代码主要覆盖三类实验：

- `FashionMNIST`：LeNet、AlexNet、GoogLeNet、VGG-16、ResNet18
- 猫狗二分类：GoogLeNetCatDot
- 口罩检测二分类：ResNet18MaskDetection

## 项目结构

```text
CNNPlayground/
├── data/                         # FashionMNIST 和 CatDog 数据目录
├── LeNet/                        # FashionMNIST: LeNet
├── AlexNet/                      # FashionMNIST: AlexNet
├── GoogLeNet/                    # FashionMNIST: GoogLeNet
├── VGG-16/                       # FashionMNIST: VGG-16
├── ResNet18/                     # FashionMNIST: ResNet18
├── GoogLeNetCatDot/              # Cat vs Dog: GoogLeNet 二分类
├── ResNet18MaskDetection/        # Mask Detection: ResNet18 二分类
├── requirements.txt
└── LICENSE
```

大多数模型目录都包含：

- `model.py`：网络结构
- `train.py`：训练、验证并保存最优权重
- `test.py`：加载 `models/best_model.pth` 后计算测试准确率
- `detect.py`：加载模型做单样本或逐样本推理
- `plot.py`：可视化一个 batch（当前多数脚本用于查看 `FashionMNIST` 样例）

图像二分类目录还包含：

- `data_partitioning.py`：将原始图片划分为 `train/` 和 `test/`
- `mean_std.py`：统计图片数据集的通道均值和方差

## 环境要求

- Python 3.10+，项目中使用过 Python 3.12
- PyTorch / torchvision
- 可选 GPU，代码会自动选择 `cuda` 或 `cpu`

安装依赖：

```bash
pip install torch torchvision
pip install -r requirements.txt
pip install sympy tqdm pillow
```

如果使用 CUDA，建议按 PyTorch 官方安装命令安装与你显卡驱动匹配的 `torch` 和 `torchvision`。

## 数据准备

### FashionMNIST

以下目录的脚本会自动下载 `FashionMNIST` 到仓库根目录的 `data/`：

- `LeNet`
- `AlexNet`
- `GoogLeNet`
- `VGG-16`
- `ResNet18`

### Cat vs Dog

`GoogLeNetCatDot` 默认使用：

```text
data/CatDog/
├── raw/
│   ├── cat/
│   └── dog/
├── train/
│   ├── cat/
│   └── dog/
└── test/
    ├── cat/
    └── dog/
```

先把原始图片放到 `data/CatDog/raw/`，再执行：

```bash
python GoogLeNetCatDot/data_partitioning.py
```

脚本会按 `9:1` 划分训练集和测试集。旧目录 `GoogLeNetCatDot/data_cat_dog/` 仍可作为兼容输入。

### Mask Detection

`ResNet18MaskDetection` 默认训练数据路径是：

```text
ResNet18MaskDetection/data/
├── train/
│   ├── mask/
│   └── no_mask/
└── test/
    ├── mask/
    └── no_mask/
```

如果使用自带划分脚本，请先把原始图片放到 `ResNet18MaskDetection/dataset/`，然后从该目录内执行：

```bash
cd ResNet18MaskDetection
python data_partitioning.py
python mean_std.py
cd ..
```

## 快速开始

以下命令默认从仓库根目录执行。

### FashionMNIST 模型

| 模型 | 训练 | 测试 | 推理 | 输入尺寸 | 默认 epoch |
| --- | --- | --- | --- | --- | --- |
| LeNet | `python LeNet/train.py` | `python LeNet/test.py` | `python LeNet/detect.py` | `28 x 28` | `100` |
| AlexNet | `python AlexNet/train.py` | `python AlexNet/test.py` | `python AlexNet/detect.py` | train: `224 x 224`, test: `227 x 227` | `20` |
| GoogLeNet | `python GoogLeNet/train.py` | `python GoogLeNet/test.py` | `python GoogLeNet/detect.py` | `227 x 227` | `20` |
| VGG-16 | `python VGG-16/train.py` | `python VGG-16/test.py` | `python VGG-16/detect.py` | `227 x 227` | `15` |
| ResNet18 | `python ResNet18/train.py` | `python ResNet18/test.py` | `python ResNet18/detect.py` | `227 x 227` | `20` |

`VGG-16` 还提供后台训练脚本：

```bash
bash VGG-16/run_train.sh
```

日志会写入 `VGG-16/logs/`。

### Cat vs Dog

```bash
python GoogLeNetCatDot/data_partitioning.py
python GoogLeNetCatDot/mean_std.py
python GoogLeNetCatDot/train.py
python GoogLeNetCatDot/test.py
python GoogLeNetCatDot/detect.py
```

说明：

- 模型输入为 RGB 三通道图片
- 图片会被 resize 到 `224 x 224`
- 默认训练轮数为 `50`
- `detect.py` 默认读取 `GoogLeNetCatDot/detect/2.webp`

### Mask Detection

```bash
cd ResNet18MaskDetection
python data_partitioning.py
python mean_std.py
cd ..

python ResNet18MaskDetection/train.py
python ResNet18MaskDetection/test.py
python ResNet18MaskDetection/detect.py
```

说明：

- 模型输入为 RGB 三通道图片
- 图片会被 resize 到 `224 x 224`
- 默认训练轮数为 `20`
- 分类类别为 `mask` 和 `no_mask`
- `detect.py` 默认读取 `ResNet18MaskDetection/data/test/mask/mask3.jpg`

## 训练输出

训练完成后，各模型目录通常会生成：

```text
models/
├── best_model.pth
└── training_curve.png
```

其中 `LeNet/train.py` 当前会保存 `best_model.pth`，训练曲线使用 `matplotlib` 弹窗显示，不写入 `training_curve.png`。

控制台会输出：

- 每个 epoch 的训练损失和验证损失
- 训练准确率和验证准确率
- `test.py` 的 `Test Accuracy`
- `detect.py` 的预测类别或预测编号

## 当前注意事项

- `requirements.txt` 只包含部分辅助依赖，首次运行请按上面的安装命令补齐依赖。
- 各模型脚本目前以学习和实验为主，参数主要写在代码中，还没有统一命令行配置。
- 自定义图像数据集需要先按目录结构准备图片，否则 `ImageFolder` 会找不到类别目录。
- 训练脚本会覆盖对应目录下的 `models/best_model.pth`，保存的是验证集准确率最高的一版权重。
- `AlexNet/train.py` 当前训练 resize 为 `224 x 224`，测试和推理 resize 为 `227 x 227`；如果训练时报全连接层输入维度不匹配，先将训练 resize 调整为 `227 x 227`。
