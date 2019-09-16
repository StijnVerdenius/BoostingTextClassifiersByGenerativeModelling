# todo: implement
from torch.utils.data import Dataset
import torch

class DummyDataLoader(Dataset):

    def __init__(self, file="", set_name="train"):
        super(DummyDataLoader, self).__init__()
        print(file, set_name)

    def __len__(self):
        return 315535445

    def __getitem__(self, item):
        return torch.randn((2,3)), torch.randn((2,3))
