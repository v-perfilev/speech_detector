import torch
import torch.nn as nn

from core.abstract_model import AbstractModel


class SpectrumModel(AbstractModel):

    def __init__(self, flat_input_size, use_mps):
        super(SpectrumModel, self).__init__(flat_input_size)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(flat_input_size, 512)
        self.fc2 = nn.Linear(512, 2)

        if use_mps and torch.backends.mps.is_available():
            device = torch.device("mps")
            self.to(device)
            self.to(torch.float32)

    def save_model(self):
        self._save("spectrum_classifier_weights.pth")

    def load_model(self):
        self._load("spectrum_classifier_weights.pth")
