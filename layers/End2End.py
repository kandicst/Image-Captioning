import torch
import torch.nn as nn
from torchvision import models
import numpy as np

from layers.Decoder import Decoder
from layers.Encoder import Encoder


class End2End(nn.Module):
    def __init__(self, enc_input, enc_output, dec_hidden, dec_output,
                 emb_dim, attn_dim, vocab, criterion, device):
        super().__init__()

        self.encoder = Encoder(enc_input, enc_output, device)
        self.decoder = Decoder(enc_output, dec_hidden, dec_output,
                               attn_dim, device, emb_dim, vocab.top_words)
        self.criterion = criterion
        self.vocab = vocab

        self.device = device
        self.to(self.device)

    def forward(self, x, captions, train=True):
        captions = captions.long().to(self.device)
        batch_size, caption_length = captions.shape[:2]
        out = torch.zeros((batch_size, caption_length))

        features = self.encoder(x)

        hidden = self.decoder.reset_state(batch_size=batch_size)
        decoder_input = [self.vocab.word_to_idx['<start>']] * batch_size
        decoder_input = torch.as_tensor(decoder_input).unsqueeze(1)

        total_loss = 0
        for i in range(caption_length):
            predictions, hidden, attn_weights = self.decoder(decoder_input, features, hidden)
            target = captions[:, i]
            max_words = torch.max(predictions, 1)[1]
            if train:
                loss = self.criterion(predictions, target.long())
                decoder_input = target.unsqueeze(-1)
            else:
                loss = self.criterion(predictions, max_words.long())
                decoder_input = max_words.unsqueeze(-1)
            total_loss += loss
            out[:, i] = max_words

        return out, total_loss

    def evaluate(self, x, captions=None, tf=False):

        features = self.encoder(x)

        hidden = self.decoder.reset_state(batch_size=1)
        decoder_input = [self.vocab.word_to_idx['<start>']]
        decoder_input = torch.as_tensor(decoder_input).unsqueeze(1)
        result = []

        for i in range(50):
            predictions, hidden, attn_weights = self.decoder(decoder_input, features, hidden)
            max_words = torch.max(predictions, 1)[1]
            print(predictions)
            word = self.vocab.idx_to_word[max_words.item()]
            result.append(word)
            if word == '<end>':
                break
            if not tf:
                decoder_input = max_words.unsqueeze(-1)
            else:
                target = captions[:, i]
                decoder_input = target.unsqueeze(-1)

        return result
