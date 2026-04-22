import copy
import os
import time

import torch
from sympy.core.random import shuffle
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from GoogLeNet.model import GoogLeNet
except ModuleNotFoundError:
    from model import GoogLeNet
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
PLOT_PATH = os.path.join(MODEL_DIR, 'training_curve.png')


def train_val_data_process():
    train_dataset = FashionMNIST(root=DATA_DIR,
                                 train=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(size=(227, 227)),
                                     transforms.ToTensor()
                                 ]),
                                 download=True)
    generator = torch.Generator().manual_seed(42)
    train_data, val_data = data.random_split(
        train_dataset,
        [
            round(0.8 * len(train_dataset)),
            round(0.2 * len(train_dataset))
        ],
        generator=generator)
    train_loader = data.DataLoader(train_data,
                                   batch_size=64,
                                   shuffle=True,
                                   num_workers=2)
    val_loader = data.DataLoader(dataset=val_data,
                                 batch_size=64,
                                 shuffle=True,
                                 num_workers=2)
    return train_loader, val_loader


def train_model_process(model, train_loader, val_loader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    # 复制当前模型参数
    best_model_weights = copy.deepcopy(model.state_dict())

    # 初始化模型参数
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    since = time.time()

    for epoch in range(num_epochs):
        epoch_since = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_loss = 0.0
        train_acc = 0
        val_loss = 0.0
        val_acc = 0
        train_num = 0  # 训练集样本数量
        val_num = 0

        train_iterator = enumerate(train_loader)
        if tqdm is not None:
            train_iterator = tqdm(
                train_iterator,
                total=len(train_loader),
                desc='Train',
                unit='it',
                leave=False
            )

        for batch_idx, (images, labels) in train_iterator:
            images = images.to(device)
            labels = labels.to(device)
            # 设置模型为训练模式
            model.train()
            output = model(images)
            # 查找每一行中最大值对应的行标
            prediction = torch.argmax(output, dim=1)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            # 根据反向传播得到的梯度信息，更新模型参数
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_acc += torch.sum(prediction == labels.data)
            train_num += images.size(0)

            if tqdm is not None:
                train_iterator.set_postfix(
                    loss=f'{loss.item():.4f}',
                    acc=f'{(train_acc / train_num).item():.4f}'
                )

        model.eval()
        with torch.no_grad():
            val_iterator = enumerate(val_loader)
            if tqdm is not None:
                val_iterator = tqdm(
                    val_iterator,
                    total=len(val_loader),
                    desc='Val',
                    unit='it',
                    leave=False
                )

            for batch_idx, (images, labels) in val_iterator:
                images = images.to(device)
                labels = labels.to(device)

                output = model(images)
                prediction = torch.argmax(output, dim=1)
                loss = criterion(output, labels)

                val_loss += loss.item() * images.size(0)
                val_acc += torch.sum(prediction == labels.data)
                val_num += images.size(0)

                if tqdm is not None:
                    val_iterator.set_postfix(
                        loss=f'{loss.item():.4f}',
                        acc=f'{(val_acc / val_num).item():.4f}'
                    )

        train_epoch_loss = train_loss / train_num
        val_epoch_loss = val_loss / val_num
        train_epoch_acc = (train_acc / train_num).item()
        val_epoch_acc = (val_acc / val_num).item()

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        train_accs.append(train_epoch_acc)
        val_accs.append(val_epoch_acc)

        print('{} Train loss: {:.4f}, Train accuracy: {:.4f}'.format(
            epoch,
            train_losses[-1],
            train_accs[-1]
        ))
        print('{} Val loss: {:.4f}, Val accuracy: {:.4f}'.format(
            epoch,
            val_losses[-1],
            val_accs[-1]
        ))

        if val_accs[-1] > best_acc:
            best_acc = val_accs[-1]
            best_model_weights = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - epoch_since
        print('Current epoch training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))

    train_process = pd.DataFrame(data={
        "epoch": list(range(num_epochs)),
        "train_loss": train_losses,
        "train_accuracy": train_accs,
        "val_loss": val_losses,
        "val_accuracy": val_accs
    })
    return train_process


def matplot_accuracy_loss(train_process):
    os.makedirs(MODEL_DIR, exist_ok=True)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(
        train_process['epoch'],
        train_process.train_loss,
        'ro-',
        label='Train Loss'
    )
    plt.plot(
        train_process['epoch'],
        train_process.val_loss,
        'bs-',
        label='Validate Loss'
    )
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(
        train_process['epoch'],
        train_process.train_accuracy,
        'ro-',
        label='Train Accuracy'
    )
    plt.plot(
        train_process['epoch'],
        train_process.val_accuracy,
        'bs-',
        label='Validate Accuracy'
    )
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Training curve saved to: {PLOT_PATH}")


if __name__ == '__main__':
    google_net = GoogLeNet()
    train_loader, val_loader = train_val_data_process()
    train_process = train_model_process(google_net, train_loader, val_loader, num_epochs=20)
    matplot_accuracy_loss(train_process)
