import torch.nn.functional as F
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, num_classes = 10):
        super(Model, self).__init__()
        self.Conv1 = nn.Conv2d(1, 64, 5) #24
        self.Pool1 = nn.MaxPool2d(2, 2) # 12
        self.Conv2 = nn.Conv2d(64,128, 5) #8
        self.Pool2 = nn.MaxPool2d(2, 2) #4
        self.Drop1 = nn.Dropout(0.3)
        self.FC1 = nn.Linear(128*4*4, 1024)
        self.Drop2 = nn.Dropout(0.3)
        self.FC2 = nn.Linear(1024, 1024)
        self.FC3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.Conv1(x))
        x = self.Pool1(x)
        x = F.relu(self.Conv2(x))
        x = self.Pool2(x)
        x = x.view(-1, 128*4*4)
        x = F.relu(self.FC1(x))
        x = F.relu(self.FC2(x))
        x = self.FC3(x)
        return x

