# CNNPlayground

一个基于 PyTorch 的 CNN 学习与实验仓库，当前主要包含以下两个方向：

- 使用 `FashionMNIST` 复现经典卷积网络的训练、测试与单样本推理
- 使用 `GoogLeNet` 做一个二分类的猫狗识别实验

仓库更偏向“练手型实验场”，目录拆分清晰，适合对比不同 CNN 结构在同一任务上的实现方式。

## 当前包含的模型

### FashionMNIST

- `LeNet`
- `AlexNet`
- `GoogLeNet`
- `VGG-16`

### Cat vs Dog

- `GoogLeNetCatDot`

## 项目结构

```text
CNNPlayground/
├── data/                      # 数据目录
├── LeNet/                     # LeNet: train / test / model
├── AlexNet/                   # AlexNet: train / test / detect / model
├── GoogLeNet/                 # GoogLeNet: train / test / detect / model
├── VGG-16/                    # VGG-16: train / test / detect / run_train.sh / model
├── GoogLeNetCatDot/           # 猫狗分类实验
└── requirements.txt
```

训练过程中生成的文件通常会保存在各模型目录下的 `models/` 中，例如：

- `best_model.pth`
- `training_curve.png`

## 环境要求

- Python 3.10+（仓库开发环境里使用过 Python 3.12）
- 建议在 Linux / macOS / WSL 下运行
- 如使用 GPU，请安装与你本机 CUDA 对应版本的 PyTorch

## 安装依赖

仓库里的 `requirements.txt` 只列出了一部分辅助依赖，实际运行还需要 `torch`、`torchvision`、`sympy`，以及部分脚本会用到可选的 `tqdm`。

建议直接执行：

```bash
pip install torch torchvision
pip install -r requirements.txt
pip install sympy tqdm
```

如果你使用 CUDA，请优先参考 PyTorch 官方安装方式替换第一行命令。

## 数据准备

### 1. FashionMNIST

`LeNet`、`AlexNet`、`GoogLeNet`、`VGG-16` 的脚本都会自动下载 `FashionMNIST` 到仓库根目录下的 `data/`。

### 2. Cat vs Dog

`GoogLeNetCatDot` 不会自动下载数据，默认读取：

```text
data/CatDog/
├── train/
│   ├── cat/
│   └── dog/
└── test/
    ├── cat/
    └── dog/
```

其中 `train/` 用于训练与验证划分，`test/` 目录可作为你自己的独立测试集。

## 快速开始

以下命令默认在仓库根目录执行。

### LeNet

```bash
python LeNet/train.py
python LeNet/test.py
```

说明：

- 输入尺寸：`28 x 28`
- 默认训练轮数：`100`
- 训练曲线当前为直接弹窗显示

### AlexNet

```bash
python AlexNet/train.py
python AlexNet/test.py
python AlexNet/detect.py
```

说明：

- 输入尺寸：训练 `224 x 224`，测试/推理 `227 x 227`
- 默认训练轮数：`20`
- 会保存最优权重与训练曲线

### GoogLeNet

```bash
python GoogLeNet/train.py
python GoogLeNet/test.py
python GoogLeNet/detect.py
```

说明：

- 输入尺寸：`227 x 227`
- 默认训练轮数：`20`
- 默认学习率：`1e-4`

### VGG-16

```bash
python VGG-16/train.py
python VGG-16/test.py
python VGG-16/detect.py
```

后台训练也可以直接用：

```bash
bash VGG-16/run_train.sh
```

说明：

- 输入尺寸：`227 x 227`
- 默认训练轮数：`15`
- `run_train.sh` 会把日志输出到 `VGG-16/logs/`

### GoogLeNetCatDot

```bash
python GoogLeNetCatDot/train.py
```

说明：

- 三通道 RGB 输入
- 输入尺寸：`224 x 224`
- 默认训练轮数：`50`
- 分类数：`2`（cat / dog）

## 脚本职责

- `model.py`：网络结构定义
- `train.py`：训练、验证、保存最优权重
- `test.py`：加载最优权重并输出整体测试准确率
- `detect.py`：逐样本输出预测类别与真实标签
- `run_train.sh`：后台启动训练并记录日志

## 输出结果

训练完成后，你通常会看到以下产物：

- `models/best_model.pth`：当前实验最优权重
- `models/training_curve.png`：训练/验证损失与精度曲线

控制台常见输出包括：

- 每个 epoch 的训练损失、验证损失
- 训练准确率、验证准确率
- 测试阶段的 `Test Accuracy`
- 推理阶段的 `prediction` 与 `label`

## 目前的已知特点

- 这是一个学习性质较强的实验仓库，代码可读性优先于工程封装
- 不同模型目录内部实现风格略有差异，适合横向对比
- `requirements.txt` 当前并不完整，首次运行请按上面的安装步骤补齐依赖
- `FashionMNIST` 实验大多会自动创建 `models/` 目录并保存结果

## 后续可继续完善

- 补充统一的命令行参数入口
- 补充实验结果表格与模型精度对比
- 增加更完整的依赖锁定
- 为猫狗分类实验补充测试与推理脚本说明

