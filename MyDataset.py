from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        print(len(self.data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        print(self.data)
        return len(self.data)
