import torch
from torch import nn
from torch.nn.utils import init
from torchsummary import summary
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )

        self.block3 = nn.Sequential(
            Inception(
                in_channels=192,
                kernel_count_1=64,
                kernel_count_2=(96, 128),
                kernel_count_3=(16, 32),
                kernel_count_4=32
            ),
            Inception(
                in_channels=256,
                kernel_count_1=128,
                kernel_count_2=(128, 192),
                kernel_count_3=(32, 96),
                kernel_count_4=64
            ),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )

        self.block4 = nn.Sequential(
            Inception(
                in_channels=480,
                kernel_count_1=192,
                kernel_count_2=(96, 208),
                kernel_count_3=(16, 48),
                kernel_count_4=64
            ),
            Inception(
                in_channels=512,
                kernel_count_1=160,
                kernel_count_2=(112, 224),
                kernel_count_3=(24, 64),
                kernel_count_4=64
            ),
            Inception(
                in_channels=512,
                kernel_count_1=128,
                kernel_count_2=(128, 256),
                kernel_count_3=(24, 64),
                kernel_count_4=64
            ),
            Inception(
                in_channels=512,
                kernel_count_1=112,
                kernel_count_2=(128, 288),
                kernel_count_3=(32, 64),
                kernel_count_4=64
            ),
            Inception(
                in_channels=528,
                kernel_count_1=256,
                kernel_count_2=(160, 320),
                kernel_count_3=(32, 128),
                kernel_count_4=128
            ),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )

        self.block5 = nn.Sequential(
            Inception(
                in_channels=832,
                kernel_count_1=256,
                kernel_count_2=(160, 320),
                kernel_count_3=(32, 128),
                kernel_count_4=128
            ),
            Inception(
                in_channels=832,
                kernel_count_1=384,
                kernel_count_2=(192, 384),
                kernel_count_3=(48, 128),
                kernel_count_4=128
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=10)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.block5(self.block4(self.block3(self.block2(self.block1(x)))))


class Inception(nn.Module):
    def __init__(self, in_channels, kernel_count_1, kernel_count_2, kernel_count_3, kernel_count_4):
        super(Inception, self).__init__()

        self.relu = nn.ReLU()

        # parallel branch 1: single 1*1 conv
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=kernel_count_1, kernel_size=(1, 1))

        # parallel branch 2: 1*1 conv and 3*3 conv
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=kernel_count_2[0], kernel_size=(1, 1))
        self.p2_2 = nn.Conv2d(in_channels=kernel_count_2[0], out_channels=kernel_count_2[1], kernel_size=(3, 3),
                              padding=1)

        # parallel branch 3: 1*1 conv and 5*5 conv
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=kernel_count_3[0], kernel_size=(1, 1))
        self.p3_2 = nn.Conv2d(in_channels=kernel_count_3[0], out_channels=kernel_count_3[1], kernel_size=(5, 5),
                              padding=2)

        # parallel branch 4: 3*3 MaxPool and 1*1 conv
        self.p4_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=kernel_count_4, kernel_size=(1, 1))

    def forward(self, x):
        p1 = self.relu(self.p1_1(x))
        p2 = self.relu(self.p2_2(self.relu(self.p2_1(x))))
        p3 = self.relu(self.p3_2(self.relu(self.p3_1(x))))
        p4 = self.relu(self.p4_2(self.p4_1(x)))

        # 在通道的维度上融合
        return torch.cat((p1, p2, p3, p4), dim=1)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = GoogLeNet().to(device)
    print(summary(net, input_size=(1, 224, 224)))
