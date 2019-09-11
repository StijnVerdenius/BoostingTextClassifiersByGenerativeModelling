# todo: implement
from torch.utils.data import Dataset


class DummyDataLoader(Dataset):

    def __init__(self, file="", set_name="train"):
        super(DummyDataLoader, self).__init__()
        print(file, set_name)
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
