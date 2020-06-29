import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class Encoder(nn.Module):

    def __init__(self, encoder_input, encoder_output):
        super().__init__()
        self.fc = nn.Linear(encoder_input, encoder_output)

    def forward(self, x):
        # print("WAAT ")
        # print(dir(self))
        x = torch.relu(self.fc(x))
        return x
