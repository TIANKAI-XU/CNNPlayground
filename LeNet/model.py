import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.sigmoid(self.conv1(x)))
        x = self.pool2(self.sigmoid(self.conv2(x)))
        x = self.flatten(x)
        return self.fc3(self.fc2(self.fc1(x)))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LeNet().to(device)
    print(summary(net, input_size=(1, 28, 28)))
