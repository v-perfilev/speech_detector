import torch.nn as nn

from core.abstract_model import AbstractModel


class SpectrogramModel(AbstractModel):

    def __init__(self, flat_input_size):
        super(SpectrogramModel, self).__init__(flat_input_size)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(flat_input_size, 256)
        self.fc2 = nn.Linear(256, 2)

    def save_model(self):
        self._save("spectrogram_classifier_weights.pth")

    def load_model(self):
        self._load("spectrogram_classifier_weights.pth")
