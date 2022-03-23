# encode: utf-8
# started on 2022/3/23 @zelo2
# finished on 2022/3/23 @zelo2

from torch.utils.data import Dataset, DataLoader

class lpkt_dataset(Dataset):
    def __init__(self, data):
        super(lpkt_dataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.data[index][-1]
        label = label[1:]
        return [self.data[index], label]




