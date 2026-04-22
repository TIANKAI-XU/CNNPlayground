import os

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import FashionMNIST, ImageFolder

try:
    from GoogLeNetCatDot.model import GoogLeNet
except ModuleNotFoundError:
    from model import GoogLeNet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'GoogLeNetCatDot/detect')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'best_model.pth')


def detect_image():
    image = Image.open(os.path.join(DATA_DIR, '2.webp'))
    normalize = transforms.Normalize(
        mean=[0.16207108, 0.15101928, 0.13847153],
        std=[0.05800501, 0.05212834, 0.04776142]
    )
    test_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    image = test_transform(image)
    # 添加批次维度 [batch, channel, height, width]
    image = image.unsqueeze(0)

    return image


def detect():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNet()
    model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    image = detect_image()
    image = image.to(device)

    with torch.no_grad():
        model.eval()
        output = model(image)
        prediction = torch.argmax(output, dim=1)
        print(prediction.item())


if __name__ == '__main__':
    detect()
