from torch.utils.data import Dataset
import torch
import ctypes
import multiprocessing as mp
import numpy as np

batch_size, w, h = 64, 64, 2048


class MyDataset(Dataset):
    def __init__(self, enc_captions, image_paths, data_dir, max_cache_size=10000, use_cache=False):
        self.enc_captions = enc_captions
        self.image_paths = image_paths
        self.data_dir = data_dir

        self.max_cache_size = max_cache_size
        self.cache = []
        self.cache_count = 0
        self.img_to_idx = {}
        self.use_cache = use_cache

        assert len(enc_captions) == len(image_paths)

    def set_use_cache(self, use_cache):
        if use_cache:
            x_img = tuple(self.cache)
            self.cache = torch.stack(x_img)
        else:
            self.cache = []
        self.use_cache = use_cache

    def load_img_from_disk(self, name):
        img = torch.load(name + '.pt')
        new_dim = np.prod(img.shape[1: -1])
        return torch.reshape(img, (new_dim, img.shape[-1]))

    def __getitem__(self, index):
        name = self.data_dir + self.image_paths[index].split('/')[-1]

        if not self.use_cache:
            img = self.load_img_from_disk(name)

            if self.cache_count < self.max_cache_size:
                # self.cache[self.cache_count] = img
                self.cache.append(img)
                self.img_to_idx[name] = self.cache_count
                self.cache_count += 1

            return img, self.enc_captions[index], name

        if name in self.img_to_idx.keys():
            idx = self.img_to_idx[name]
            x = self.cache[idx]
        else:
            x = self.load_img_from_disk(name)

        return x, self.enc_captions[index], name

    def __len__(self):
        return len(self.enc_captions)


if __name__ == '__main__':
    pass
