import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class Encoder(nn.Module):

    def __init__(self, encoder_input, encoder_output, device):
        super().__init__()
        self.fc = nn.Linear(encoder_input, encoder_output)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = torch.relu(self.fc(x))
        return x
