import torch
import torch.nn as nn


class Attention(nn.Module):
    """ Attention mechanism of decoder
    Parameters
    -----------
    encoder_dim: int
        size of decoder output
    decoder_dim: int
        size of hidden state
    attention_dim: int
        size of attention output
    device : torch.device
        device to send data to

    Attributes
    -----------
    features_fc: torch.nn.Linear
        fully connected layer for image vector
    hidden_fc: torch.nn.Linear
        fully connected layer for RNN hidden state
    combined_fc: torch.nn.Linear
        fully connected layer to combine previous two
    softmax: torch.nn.Softmax
        used for normalizing attention to sum up to one
    """
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
        """ Forward pass """
        features = features.to(self.device)

        h_t = hidden.unsqueeze(1)

        a = self.features_fc(features)
        h_t = self.hidden_fc(h_t)

        e_t = torch.tanh(a + h_t)
        full = self.combined_fc(e_t)
        alpha = self.softmax(full)
        context_vector = (features * alpha).sum(dim=1)

        return context_vector, alpha
