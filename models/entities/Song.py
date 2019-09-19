import re
import sys

sys.path.append('..')

from models.enums.Genre import Genre

def remove_ending_substring(string, substring):
    while string.endswith(substring):
        return string[:-len(substring)]

    return string

class Song():

    def __init__(self, genre: Genre, lyrics: str):

        self._genre = genre
        
        parsed_lyrics = re.sub('\n+', '\n', lyrics)
        parsed_lyrics = remove_ending_substring(parsed_lyrics, '\n')
        parsed_lyrics = parsed_lyrics.replace('\n \n', '\n')

        self._lyrics = parsed_lyrics
        self._start_index = 0

    def __str__(self):
        result = f'Song:\n - Genre: "{self.genre}"\n - Start index: "{self.start_index}"\n - Number of lines: "{self.number_of_lines}"'
        return result

    @property
    def genre(self) -> Genre:
        return self._genre
        
    @property
    def lyrics(self) -> str:
        return self._lyrics
        
    @property
    def start_index(self) -> int:
        return self._start_index

    @start_index.setter
    def start_index(self, start_index):
        self._start_index = start_index
        
    @property
    def number_of_lines(self) -> int:
        return self._lyrics.count('\n') + 1