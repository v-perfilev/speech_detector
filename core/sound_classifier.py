import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoundClassifier(nn.Module):
    flat_input_size = 0

    def __init__(self, flat_input_size):
        super(SoundClassifier, self).__init__()
        self.flat_input_size = flat_input_size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(self.flat_input_size, 64)
        self.fc2 = nn.Linear(64, 2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.flat_input_size)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save_model(self, dict_path="target/sound_classifier_weights.pth"):
        os.makedirs('target', exist_ok=True)
        torch.save(self.state_dict(), dict_path)
        print('Model saved')

    def load_model(self, dict_path="target/sound_classifier_weights.pth"):
        self.load_state_dict(torch.load(dict_path))
        self.eval()
