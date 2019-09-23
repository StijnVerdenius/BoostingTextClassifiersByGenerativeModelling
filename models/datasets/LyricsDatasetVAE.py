from typing import List

from models.datasets.LyricsDataset import LyricsDataset
from models.entities.Song import Song
from models.enums.Genre import Genre
from utils.data_manager import DataManager


class LyricsDatasetVAE(LyricsDataset):

    def __init__(self, folder, set_name, genre: Genre = None, **kwargs):
        self.genre = genre
        super().__init__(folder, set_name, **kwargs)

    def setup(self, data_manager: DataManager, set_name: str):
        x = data_manager.load_python_obj(f'song_lyrics.{set_name}')
        x: List[Song]
        return [song for song in x if self.genre == song.genre]
