import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractModel(nn.Module):
    flat_input_size = None

    def __init__(self, flat_input_size):
        super(AbstractModel, self).__init__()
        self.flat_input_size = flat_input_size

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
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

    def _save(self, filename, dict_path="target"):
        os.makedirs(dict_path, exist_ok=True)
        torch.save(self.state_dict(), dict_path + "/" + filename)

    def _load(self, filename, dict_path="target"):
        self.load_state_dict(torch.load(dict_path + "/" + filename))
        self.eval()
