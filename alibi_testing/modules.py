"""
Modules for PyTorch models, required at runtime to load persisted models.
"""
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, padding='same')
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, padding='same')
        self.fc1 = nn.Linear(1568, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.dropout(F.max_pool2d(F.relu(self.conv1(x)), 2), p=0.3, training=self.training)
        x = F.dropout(F.max_pool2d(F.relu(self.conv2(x)), 2), p=0.3, training=self.training)
        x = x.reshape(-1, 1568)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
        x = self.fc2(x)
        return x
