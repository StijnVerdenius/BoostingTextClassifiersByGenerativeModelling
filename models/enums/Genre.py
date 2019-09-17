from enum import Enum

class Genre(Enum):
    Pop = 0
    HipHop = 1
    Rock = 2
    Metal = 3
    Country = 4

    @classmethod
    def from_str(cls, label):
        if label == 'Pop':
            return cls.Pop
        elif label == 'Hip-Hop':
            return cls.HipHop
        elif label == 'Rock':
            return cls.Rock
        elif label == 'Metal':
            return cls.Metal
        elif label == 'Country':
            return cls.Country

        return None