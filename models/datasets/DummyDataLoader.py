# todo: implement
from torch.utils.data import Dataset
import torch

class DummyDataset(Dataset):

    def __init__(self, file="", set_name="train"):
        super(DummyDataset, self).__init__()
        print(file, set_name)

    def __len__(self):
        return 3150

    def __getitem__(self, item):
        return torch.randn((2,100)), torch.randn((100))
