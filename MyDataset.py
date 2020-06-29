from torch.utils.data import Dataset
import torch
import numpy as np

class MyDataset(Dataset):
    def __init__(self, enc_captions, image_paths, data_dir):
        self.enc_captions = enc_captions
        self.image_paths = image_paths
        self.data_dir = data_dir

        assert len(enc_captions) == len(image_paths)

    def __getitem__(self, index):
        string = self.data_dir + self.image_paths[index].split('/')[-1]
        img = torch.load(string + '.pt')
        new_dim = np.prod(img.shape[1:-1])
        img = torch.reshape(img, (new_dim, img.shape[-1]))
        return img, self.enc_captions[index]

    def __len__(self):
        return len(self.enc_captions)


if __name__ == '__main__':
    pass