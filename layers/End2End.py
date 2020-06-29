import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class End2End(nn.Module):
    def __init__(self, encoder, decoder, criterion, device):
        super(End2End, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.to(device)

    def forward(self, x, captions):
        batch_size, caption_length = x.shape[:2]

        features = self.encoder(x)

        hidden = self.decoder.reset_state(batch_size=batch_size)
        decoder_input = [word_to_idx['<start>']] * batch_size
        decoder_input = torch.as_tensor(decoder_input).unsqueeze(1)

        loss = 0
        for i in range(caption_length):
            predictions, hidden = self.decoder(decoder_input, features, hidden)
            loss += self.criterion(captions[:,i].unsqueeze(-1), predictions)