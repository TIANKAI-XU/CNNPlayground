import os

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import FashionMNIST, ImageFolder

from ResNet18MaskDetection.test import test_val_data_process

try:
    from ResNet18MaskDetection.model import ResNet18MaskDetection
except ModuleNotFoundError:
    from model import ResNet18MaskDetection

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'ResNet18MaskDetection/data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'best_model.pth')


def detect_image():
    image = Image.open(os.path.join(TEST_DIR, 'mask/mask3.jpg'))

    test_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
    ])
    image = test_transform(image)
    # 添加批次维度 [batch, channel, height, width]
    image = image.unsqueeze(0)

    return image


def detect():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18MaskDetection()
    model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    img = detect_image().to(device)

    with torch.no_grad():
        model.eval()

        output = model(img)
        prediction = torch.argmax(output, dim=1)

        class_names = ["mask", "no_mask"]
        predicted_class = class_names[prediction.item()]

        print("prediction: {}".format(predicted_class))


if __name__ == '__main__':
    detect()
