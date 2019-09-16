# todo: implement
from torch.utils.data import Dataset
import torch

class DummyDataLoader(Dataset):

    def __init__(self, file="", set_name="train"):
        super(DummyDataLoader, self).__init__()
        print(file, set_name)
        # raise NotImplementedError

    def __len__(self):
        return 10
        # raise NotImplementedError

    def __getitem__(self, item):
        return torch.randn((2,3))
        # raise NotImplementedError
