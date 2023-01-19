import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3_drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(4 * 4 * 64, 256)
        self.fc2 = nn.Linear(256, 10)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.fc1_drop(x)
        x = self.act(self.conv2(x))
        x = self.fc2_drop(x)
        x = self.act(self.conv3(x))
        x = self.fc3_drop(x)
        x = self.act(self.conv4(x))
        x = x.view(-1, 4 * 4 * 64)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

