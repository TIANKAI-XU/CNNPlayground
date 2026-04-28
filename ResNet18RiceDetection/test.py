import os

import torch.cuda
from torch import nn
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import FashionMNIST, ImageFolder

try:
    from ResNet18RiceDetection.model import ResNet18RiceDetection
except ModuleNotFoundError:
    from model import ResNet18RiceDetection

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'ResNet18RiceDetection/data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'best_model.pth')


def test_val_data_process():
    normalize = transforms.Normalize(
        mean=[0.04206815, 0.04280969, 0.04413873],
        std=[0.03315933, 0.034333, 0.03628065]
    )
    test_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = ImageFolder(root=TEST_DIR, transform=test_transform)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    return test_loader


def test_model_process(model, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    test_accs = 0.0
    test_num = 0

    class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            model.eval()
            output = model(image)
            prediction = torch.argmax(output, dim=1)  # 返回最大值对应的下标

            predicted_class = class_names[prediction.item()]
            true_class = class_names[label.item()]

            test_accs += torch.sum(prediction == label.data)
            test_num += image.size(0)

            print("prediction: {}, label: {}".format(predicted_class, true_class))

    test_acc = test_accs / test_num
    print("Test Accuracy: {:.4f}".format(test_acc))


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet18RiceDetection()
    model = model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    test_loader = test_val_data_process()
    test_model_process(model, test_loader)


if __name__ == '__main__':
    test()
