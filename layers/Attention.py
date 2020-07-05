import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim, device):
        super().__init__()
        self.device = device
        self.attention_dim = attention_dim

        self.features_fc = nn.Linear(encoder_dim, attention_dim)
        self.hidden_fc = nn.Linear(decoder_dim, attention_dim)
        self.combined_fc = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)

        self.to(device)

    def forward(self, features, hidden):
        features = features.to(self.device)

        h_t = hidden.unsqueeze(1)

        a = self.features_fc(features)
        h_t = self.hidden_fc(h_t)

        e_t = torch.tanh(a + h_t)
        # print("SCORE: ", e_t.shape)
        full = self.combined_fc(e_t)
        # print(full.shape)
        alpha = self.softmax(full)
        # print("ATT: ", alpha.shape)
        # print(alpha[0])
        # print(alpha.shape)
        # print(alpha[0].tolist())
        context_vector = (features * alpha).sum(dim=1)
        # print("CTX: ", context_vector.shape)
        # raise Exception()

        return context_vector, alpha
