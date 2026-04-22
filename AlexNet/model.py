import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 96, 11, 4)
        self.pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.pool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(3, 2)

        self.fc1 = nn.Linear(in_features=6 * 6 * 256, out_features=4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool3(torch.relu(self.conv5(x)))

        x = nn.Flatten()(x)

        x = F.dropout(torch.relu(self.fc1(x)), 0.5)
        x = F.dropout(torch.relu(self.fc2(x)), 0.5)
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = AlexNet().to(device)
    print(summary(net, input_size=(1, 255, 255)))
