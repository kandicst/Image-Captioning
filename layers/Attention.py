import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        self.features_fc = nn.Linear(encoder_dim, attention_dim)
        self.hidden_fc = nn.Linear(decoder_dim, attention_dim)
        self.combined_fc = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, hidden):

        a = self.features_fc(features)
        h_t = self.hidden_fc(hidden)

        e_t = self.combined_fc(self.relu(a + h_t))
        alpha = self.softmax(e_t)
        context_vector = (features * alpha).sum(dim=1)

        return context_vector, alpha