import torch
import torch.nn as nn


class Model_1(nn.Module):

    def __init__(self, num_classes=10):
        super(Model_1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            # 特征图大小：16@26*26
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 特征图大小：16@13*13
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*13*13, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class Model_2(nn.Module):

    def __init__(self, num_classes=10):
        super(Model_2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            # 特征图大小：16@26*26
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 特征图大小：16@13*13
            nn.Conv2d(16, 32, kernel_size=3),
            # 特征图大小：32@11*11
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 特征图大小：32@5*5
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*5*5, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class Model_3(nn.Module):

    def __init__(self, num_classes=10):
        super(Model_3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            # 特征图大小：16@26*26
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 特征图大小：16@13*13
            nn.Conv2d(16, 32, kernel_size=3),
            # 特征图大小：32@11*11
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 特征图大小：32@5*5
            nn.Conv2d(32, 48, kernel_size=3),
            # 特征图大小：48@3*3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 特征图大小：48@1*1
        )
        self.classifier = nn.Sequential(
            nn.Linear(48*1*1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x