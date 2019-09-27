from torch.utils.data import Dataset
from utils.constants import *
from utils.model_utils import find_right_model
from models.datasets.BaseDataset import BaseDataset
from torch.utils.data import DataLoader
from models.enums.Genre import Genre
from utils.dataloader_utils import pad_and_sort_batch


class WrapperLoader(Dataset):

    def __init__(self, arguments, folder, set_name, genre, normalize):
        arguments.dataset_class = 'LyricsDataset'
        self.data_loader_test = self.load_dataloader2(arguments, TEST_SET)
        arguments.dataset_class = 'LyricsRawDataset'
        self.data_loader_sentenceVAE = self.load_dataloader2(arguments, TEST_SET)

        self.sos_idx = self.data_loader_sentenceVAE.dataset.sos_idx
        self.eos_idx = self.data_loader_sentenceVAE.dataset.eos_idx
        self.pad_idx = self.data_loader_sentenceVAE.dataset.pad_idx
        self.unk_idx = self.data_loader_sentenceVAE.dataset.unk_idx
        self.vocab_size = self.data_loader_sentenceVAE.dataset.vocab_size

    def __len__(self):
        return self.data_loader_test.dataset.__len__()

    def __getitem__(self, idx):
        print(self.data_loader_test.dataset.__len__())
        print(self.data_loader_sentenceVAE.dataset.__len__())

        return self.data_loader_test.dataset.__getitem__(idx), self.data_loader_sentenceVAE.dataset.__getitem__(idx)

    def load_dataloader2(self, arguments, set_name: str) -> DataLoader:
        """ loads specific dataset as a DataLoader """

        dataset: BaseDataset = find_right_model(DATASETS,
                                                arguments.dataset_class,
                                                folder=arguments.data_folder,
                                                set_name=set_name,
                                                genre=Genre.from_str(arguments.genre),
                                                normalize=arguments.normalize_data)

        loader = DataLoader(
            dataset,
            shuffle=(set_name is TRAIN_SET),
            batch_size=arguments.batch_size)

        if dataset.use_collate_function():
            loader.collate_fn = pad_and_sort_batch

        return loader

    def use_collate_function(self) -> bool:
        return False