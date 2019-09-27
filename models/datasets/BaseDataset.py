from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        super(BaseDataset, self).__init__()

    def use_collate_function(self) -> bool:
        return False