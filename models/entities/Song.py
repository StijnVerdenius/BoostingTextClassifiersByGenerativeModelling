import re

def remove_ending_substring(string, substring):
    while string.endswith(substring):
        return string[:-len(substring)]

    return string

class Song():

    def __init__(self, genre, lyrics, start_index):

        self._genre = genre
        
        parsed_lyrics = re.sub('\n+', '\n', lyrics)
        parsed_lyrics = remove_ending_substring(parsed_lyrics, '\n')
        self._lyrics = parsed_lyrics
        self._start_index = start_index

    def __str__(self):
        result = f'Song:\n - Genre: "{self.genre}"\n - Start index: "{self.start_index}"\n - Number of lines: "{self.number_of_lines}"'
        return result

    @property
    def genre(self) -> str:
        return self._genre
        
    @property
    def lyrics(self) -> str:
        return self._lyrics
        
    @property
    def start_index(self) -> str:
        return self._start_index
        
    @property
    def number_of_lines(self) -> str:
        return self._lyrics.count('\n') + 1