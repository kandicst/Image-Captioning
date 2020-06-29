import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class End2End(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(End2End, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.to(device)

    def forward(self, x, captions):
        print(x.shape)
        batch_size = x.shape[1]
        encoder_out = self.encoder(x)
        print(encoder_out.shape)
        max_len = captions.shape[0]

        all_outputs = torch.zeros(max_len, batch_size, self.decoder.decoder_dims)

        # first input yo yhe decoder is <START> token
        out = captions[0, :]
        print("START " + str(out))

        hidden = torch.zeros(self.decoder.encoder_dims)
        for i in range(max_len):
            out, hidden = self.decoder(out, hidden, encoder_out)
            all_outputs[i] = out
            out = out.max(1)[1]

        return all_outputs
