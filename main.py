from utils import *
from Vocabulary import Vocabulary
from layers.End2End import End2End
import time
from PIL import Image
from data_preprocessing import *
import urllib.request
import io
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader
from MyDataset import MyDataset
from nltk.translate.bleu_score import corpus_bleu
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extracted_data_train_dir = 'extracted_data/train/'
extracted_data_test_dir = 'extracted_data/test/'

train_captions = load_json(extracted_data_train_dir + 'captions.json')
train_image_paths = load_json(extracted_data_train_dir + 'image_paths.json')

EPOCHS = 500
max_vocab_size = 5000
DEC_EMB_DIM = 512
ENC_INPUT = 2048
ENC_OUTPUT = 256
DEC_HID_DIM = 512
DEC_OUTPUT = 512
ATTN_DIM = 512
EMB_DIM = 256
PAD_IDX = 0
TRAIN_SIZE = 25600
loss_plot = []

torch.autograd.set_detect_anomaly(True)


def get_model(name='saved_models/model', path=extracted_data_train_dir, size=TRAIN_SIZE):
    train_vocab = Vocabulary(train_captions, max_vocab_size)
    wat = [torch.tensor(x[1:], dtype=torch.int16) for x in train_vocab.encoded_captions]
    padded = pad_sequence(wat).permute(1, 0)

    dataset = MyDataset(enc_captions=padded[:size],
                        image_paths=train_image_paths[:size],
                        data_dir=path + 'vecs/')

    dataloader = DataLoader(dataset=dataset, batch_size=256,
                            num_workers=0)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    model = End2End(ENC_INPUT, ENC_OUTPUT, DEC_HID_DIM, DEC_OUTPUT,
                    EMB_DIM, ATTN_DIM, train_vocab, criterion, device)

    model.load_state_dict(torch.load(name))

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer.lr = 0.001

    return model, dataset, dataloader, optimizer


def main():
    model, dataset, dataloader, optimizer = get_model()
    train(model, dataset, dataloader, optimizer)


def train(model, dataset, dataloader, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0

        steps = 0
        total_bleu = 0
        for idx, batch in enumerate(dataloader):
            img_tensor, target, img_names = batch[0], batch[1], batch[2]

            out, batch_loss, t_loss = train_step(img_tensor, target, model, optimizer)

            batch_bleu = calculate_bleu(out, img_names, dataset, model.vocab)
            total_bleu += batch_bleu

            total_loss += t_loss.item()

            if idx % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Bleu {:.4f}'.format(
                    epoch + 1, idx, batch_loss.item() / int(target.shape[1]), batch_bleu))
            steps += 1

        if epoch == 0:
            dataloader.dataset.set_use_cache(True)
            dataloader.num_workers = 4

        torch.save(model.state_dict(), 'saved_models/model')

        print('Epoch {} Loss {:.6f} Bleu {:.4f}'.format(epoch + 1,
                                                        total_loss / steps, total_bleu / steps))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def train_step(batch, captions, model, optimizer):
    batch_size = captions.shape[0]
    caption_length = captions.shape[1]

    optimizer.zero_grad()
    out, loss = model(batch, captions)

    loss.backward()
    optimizer.step()
    total_loss = loss / int(caption_length)
    return out, loss, total_loss


if __name__ == '__main__':
    main()
