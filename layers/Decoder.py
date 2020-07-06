import torch
import torch.nn as nn
from layers.Attention import Attention


class Decoder(nn.Module):
    """ Network for decoding image to sentence
    Parameters
    -----------
    encoder_dim: int
        size of decoder output
    decoder_hidden_dim: int
        size of decoder output
    decoder_dim: int
        size of hidden state
    attention_dim: int
        size of attention output
    device : torch.device
        device to send data to
    embedding_dim: int
        number of dimension to embed captions in
    vocab_size: int
        number of words in vocabulary
    dropout: float
        dropout percentage

    Attributes
    -----------
    embedding: torch.nn.Embedding
        embedding layer for captions
    attention: Attention
        attention mechanism of decoder
    rnn: torch.nn.LSTM
        recurrent neural network for generating words
    fc1, fc2: torch.nn.Linear
        fully connected layers for output of decoder
    """

    def __init__(self, encoder_dim, decoder_hidden_dim, decoder_dim, attention_dim, device,
                 embedding_dim=256, vocab_size=10000, dropout_p=0.5):
        super().__init__()

        self.encoder_dims = encoder_dim
        self.decoder_dims = decoder_dim
        self.attention_dim = attention_dim
        self.vocab_size = vocab_size
        self.decoder_hidden = decoder_hidden_dim
        self.embedding_dim = embedding_dim
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(encoder_dim, decoder_hidden_dim, attention_dim, device)
        self.rnn = nn.LSTM(input_size=encoder_dim + embedding_dim, hidden_size=decoder_hidden_dim)

        self.fc1 = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.fc2 = nn.Linear(decoder_hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_p)

        self.to(device)

    def forward(self, x, features, hidden):
        """ Forward pass """
        x = x.long().to(self.device)
        hidden = hidden.to(self.device)
        context_vector, attention_weights = self.attention(features, hidden)
        context_vector = context_vector.unsqueeze(1)

        x = self.embedding(x)

        x = torch.cat([x, context_vector], dim=-1)

        output, h = self.rnn(x)
        state, cell = h
        state = state.reshape(state.shape[1:])

        x = self.fc1(output)
        x = x.reshape((-1, x.shape[2]))
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        """ Returns inial state of RNN (zeros)"""
        return torch.zeros((batch_size, self.decoder_dims))
