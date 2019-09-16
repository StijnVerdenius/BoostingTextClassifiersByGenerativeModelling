import os
import h5py

from utils.data_manager import DataManager

from torch.utils.data import Dataset
import torch

class LyricsDataset(Dataset):

    def __init__(self, folder, set_name, **kwargs):
        super(LyricsDataset, self).__init__()
        
        data_manager = DataManager(folder)
        self._song_entries = data_manager.load_python_obj(f'song_lyrics.{set_name}')

        embeddings_folder_path = os.path.join(folder, 'embeddings')
        assert os.path.exists(embeddings_folder_path)
        self._embeddings_file_path = os.path.join(embeddings_folder_path, f'embeddings.{set_name}.hdf5')
        assert os.path.exists(self._embeddings_file_path)


    def __len__(self):
        return len(self._song_entries)

    def __getitem__(self, index):
        
        song_entry = self._song_entries[index]
        
        embeddings_file = h5py.File(self._embeddings_file_path, 'r')
        embeddings = []

        for index in range(song_entry.start_index, song_entry.start_index + song_entry.number_of_lines):
            sentence_elmo_embeddings = torch.Tensor(embeddings_file[str(index)])
            embeddings.append(sentence_elmo_embeddings)

        embeddings_file.close()

        return embeddings, song_entry.genre