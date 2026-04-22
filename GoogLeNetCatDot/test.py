import os

import torch.cuda
from torch import nn
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import FashionMNIST, ImageFolder

try:
    from GoogLeNetCatDot.model import GoogLeNet
except ModuleNotFoundError:
    from model import GoogLeNet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DATASET_DIR = os.path.join(DATA_DIR, 'CatDog')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'best_model.pth')


def test_val_data_process():
    normalize = transforms.Normalize(
        mean=[0.16207108, 0.15101928, 0.13847153],
        std=[0.05800501, 0.05212834, 0.04776142]
    )
    test_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = ImageFolder(root=TEST_DIR, transform=test_transform)

    test_loader = data.DataLoader(test_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=0)

    return test_loader


def test_model_process(model, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    test_accs = 0.0
    test_num = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            model.eval()
            output = model(image)
            prediction = torch.argmax(output, dim=1)  # 返回最大值对应的下标

            test_accs += torch.sum(prediction == label.data)
            test_num += image.size(0)

    test_acc = test_accs / test_num
    print("Test Accuracy: {:.4f}".format(test_acc))


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GoogLeNet()
    model = model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    test_loader = test_val_data_process()
    test_model_process(model, test_loader)


if __name__ == '__main__':
    test()
