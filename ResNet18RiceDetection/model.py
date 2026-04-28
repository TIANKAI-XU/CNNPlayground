import torch
from torch import nn
from torch.nn.utils import init
from torchsummary import summary
import torch.nn.functional as F


class ResNet18RiceDetection(nn.Module):
    def __init__(self):
        super(ResNet18RiceDetection, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )

        self.block2 = nn.Sequential(
            Residual(input_channels=64, num_kernels=64, use_1Mul1=False, strides=1),
            Residual(input_channels=64, num_kernels=64, use_1Mul1=False, strides=1)
        )

        self.block3 = nn.Sequential(
            Residual(input_channels=64, num_kernels=128, use_1Mul1=True, strides=2),
            Residual(input_channels=128, num_kernels=128, use_1Mul1=False, strides=1)
        )

        self.block4 = nn.Sequential(
            Residual(input_channels=128, num_kernels=256, use_1Mul1=True, strides=2),
            Residual(input_channels=256, num_kernels=256, use_1Mul1=False, strides=1)
        )

        self.block5 = nn.Sequential(
            Residual(input_channels=256, num_kernels=512, use_1Mul1=True, strides=2),
            Residual(input_channels=512, num_kernels=512, use_1Mul1=False, strides=1)
        )

        self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=5)
        )

    def forward(self, x):
        return self.block6(self.block5(self.block4(self.block3(self.block2(self.block1(x))))))


class Residual(nn.Module):
    def __init__(self, input_channels, num_kernels, use_1Mul1=False, strides=1):
        super(Residual, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=num_kernels, kernel_size=(3, 3), padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=num_kernels)
        self.bn2 = nn.BatchNorm2d(num_features=num_kernels)
        if use_1Mul1:
            self.conv3 = nn.Conv2d(
                in_channels=input_channels, out_channels=num_kernels, kernel_size=(1, 1), stride=strides
            )
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        return self.relu(x + y)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ResNet18RiceDetection().to(device)
    print(summary(net, input_size=(1, 224, 224)))
