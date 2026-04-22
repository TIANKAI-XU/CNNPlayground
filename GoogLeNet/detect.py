import os

import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

try:
    from GoogLeNet.model import GoogLeNet
except ModuleNotFoundError:
    from model import GoogLeNet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'best_model.pth')


def test_val_data_process():
    test_dataset = FashionMNIST(root=DATA_DIR,
                                train=False,
                                transform=transforms.Compose([
                                    transforms.Resize(size=(227, 227)),
                                    transforms.ToTensor()
                                ]),
                                download=True)

    test_loader = data.DataLoader(test_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=0)

    return test_loader


def detect():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNet()
    model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_loader = test_val_data_process()
    class_names = test_loader.dataset.classes

    with torch.no_grad():
        model.eval()

        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            prediction = torch.argmax(output, dim=1)

            predicted_class = class_names[prediction.item()]
            true_class = class_names[label.item()]

            print("prediction: {}, label: {}".format(predicted_class, true_class))


if __name__ == '__main__':
    detect()
