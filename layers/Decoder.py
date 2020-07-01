import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from layers.Attention import Attention


class Decoder(nn.Module):
    def __init__(self, encoder_dim, decoder_hidden_dim, decoder_dim, attention_dim, device,
                 embedding_dim=256, vocab_size=10000, dropout_p=0.5):
        super().__init__()

        self.encoder_dims = encoder_dim
        self.decoder_dims = decoder_dim
        self.attention_dim = attention_dim
        self.vocab_size = vocab_size
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim, device)
        self.rnn = nn.GRU(encoder_dim + embedding_dim, decoder_hidden_dim, batch_first=True)

        self.fc1 = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.fc2 = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_p)

        self.to(device)

    def forward(self, x, features, hidden):
        x = x.long().to(self.device)
        context_vector, attention_weights = self.attention(features, hidden)

        x = self.embedding(x)

        context_vector = context_vector.unsqueeze(1)
        x = torch.cat([x, context_vector], dim=-1)

        output, state = self.rnn(x)
        state = state.reshape(state.shape[1:])

        x = self.fc1(x)
        x = x.reshape((-1, x.shape[2]))
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return torch.zeros((batch_size, self.decoder_dims))
