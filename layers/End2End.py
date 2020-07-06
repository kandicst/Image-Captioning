import torch
import torch.nn as nn
from layers.Decoder import Decoder
from layers.Encoder import Encoder


class End2End(nn.Module):
    """ Network that encapsulates encoder and decoder into one"""

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

    def forward(self, x, captions):
        """ Forward pass """
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
            max_words = torch.argmax(predictions, 1)
            loss = self.criterion(predictions, target.long())
            # loss += ((1. - attn_weights.sum(dim=1)) ** 2).mean()
            decoder_input = target.unsqueeze(-1)
            total_loss += loss
            out[:, i] = max_words

        return out, total_loss

    def evaluate(self, x):
        """ Evaluates single input """
        features = self.encoder(x)

        hidden = self.decoder.reset_state(batch_size=1)
        decoder_input = [self.vocab.word_to_idx['<start>']]
        decoder_input = torch.as_tensor(decoder_input).unsqueeze(1)
        result = []
        attention_weights = []

        with torch.no_grad():

            for i in range(20):
                predictions, hidden, attn_weights = self.decoder(decoder_input, features, hidden)
                attn_weights = attn_weights.reshape(attn_weights.shape[1])
                max_words = torch.max(predictions, 1)[1]
                word = self.vocab.idx_to_word[max_words.item()]
                result.append(word)
                attention_weights.append(attn_weights)
                if word == '<end>':
                    break
                decoder_input = max_words.unsqueeze(-1)

        return result, attention_weights
