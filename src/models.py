import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DeeperPneumoniaCNN(nn.Module):
    def __init__(self):
        super(DeeperPneumoniaCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout1 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(256, 128)

        self.dropout2 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.dropout1(x)
        x = F.relu(self.fc1(x))

        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def build_resnet18(pretrained=True):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    return model


def get_model(config, pretrained=True):
    model_name = config["model"]

    if model_name == "cnn":
        return PneumoniaCNN()
    elif model_name == "deeper":
        return DeeperPneumoniaCNN()
    elif model_name == "resnet18":
        return build_resnet18(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
