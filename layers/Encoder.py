import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, x):
        # x = self.fc(x)
        return x
