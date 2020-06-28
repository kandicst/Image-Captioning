import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from layers.Attention import Attention


class Decoder(nn.Module):
    def __init__(self, decoder_hidden_dim, decoder_dim, attention_dim, encoder_dim=2048,
                 embedding_dim=1000, dropout_p=0.5, vocab_size=10000):
        super(Decoder, self).__init__()

        self.encoder_dims = encoder_dim
        self.decoder_dims = decoder_dim
        self.attention_dim = attention_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.rnn = nn.GRU(encoder_dim + embedding_dim, decoder_hidden_dim)

        self.fc1 = nn.Linear(attention_dim + embedding_dim, decoder_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)

        x = self.embedding(x)

        x = torch.cat([x, context_vector], dim=2)

        x, hidden = self.rnn(x, hidden.unsqueeze(0))

        x = self.fc1(x)

        return x, hidden
