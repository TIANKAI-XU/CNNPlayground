import os

from sympy.codegen.fnodes import size
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


def main():
    train_dataset = FashionMNIST(root=DATA_DIR,
                                 train=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(size=224),
                                     transforms.ToTensor()
                                 ]),
                                 download=True)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=64,
                                   shuffle=True,
                                   num_workers=0)

    for step, (b_x, b_y) in enumerate(train_loader):
        if step > 0:
            break

    batch_x = b_x.squeeze().numpy()
    batch_y = b_y.numpy()
    class_label = train_dataset.classes
    # print(class_label)

    # 可视化一个batch的图像
    plt.figure(figsize=(12, 5))
    for i in np.arange(len(batch_y)):
        plt.subplot(4, 16, i + 1)
        plt.imshow(batch_x[i, :, :], cmap=plt.cm.gray)
        plt.title(class_label[batch_y[i]], size=10)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)
    plt.show()


if __name__ == '__main__':
    main()
