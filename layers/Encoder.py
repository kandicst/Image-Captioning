import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class Encoder(nn.Module):
    """ Network for encoding an image to vector
    Parameters
    -----------
    encoder_input: int
        size of encoder input
    encoder_output: int
        size of encoder output
    device : torch.device
        device to send data to

    Attributes
    -----------
    fc: torch.nn.Linear
        fully connected layers for output of encoder
    """

    def __init__(self, encoder_input, encoder_output, device):
        super().__init__()
        self.fc = nn.Linear(encoder_input, encoder_output)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = torch.relu(self.fc(x))
        return x
