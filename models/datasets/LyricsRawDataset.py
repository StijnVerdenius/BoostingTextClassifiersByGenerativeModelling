import os
import io
import json
import torch
import numpy as np
from typing import List, Tuple
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize

from utils.constants import *
from utils.dataloader_utils import OrderedCounter
from utils.data_manager import DataManager

from models.entities.Song import Song
from models.enums.Genre import Genre
from models.datasets.BaseDataset import BaseDataset

class LyricsRawDataset(BaseDataset):

    def __init__(self, folder, set_name, create_data=False, genre: Genre = None, **kwargs):

        super().__init__()
        self.data_dir = folder
        self.split = set_name
        self.max_sequence_length = kwargs.get('max_sequence_length', 500)
        self.min_occ = kwargs.get('min_occ', 3)
        self.genre = genre

        data_manager = DataManager(folder)

        # load the song entries pickle
        song_entries = self.setup(data_manager, set_name)

        self.data_file = f'lyrics.{set_name}.{genre}.json'
        self.vocab_file = f'lyrics.vocab.json'

        if create_data:
            print("Creating new %s ptb data."%set_name.upper())
            self._create_data(song_entries)

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(set_name.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data(song_entries)

        else:
            self._load_data()

    def use_collate_function(self) -> bool:
        return False

    def setup(self, data_manager: DataManager, set_name: str) -> List[Song]:
        x: List[Song] = data_manager.load_python_obj(f'song_lyrics.{set_name}')
        return x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return (np.asarray(self.data[idx]['input'], dtype=np.int64), np.asarray(self.data[idx]['target'], dtype=np.int64), self.data[idx]['length'])


    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):

        with open(os.path.join(self.data_dir, self.data_file), 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r', encoding='utf-8') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r', encoding='utf-8') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self, song_entries):

        if self.split == TRAIN_SET and not os.path.exists(os.path.join(self.data_dir, self.vocab_file)):
            self._create_vocab(song_entries)
        else:
            self._load_vocab()

        data = defaultdict(dict)
        # with open(self.raw_data_path, 'r', encoding='utf-8') as file:

        for song_entry in song_entries:
            if song_entry.genre != self.genre and self.genre != None:
                continue

            words = word_tokenize(song_entry.lyrics)

            input = ['<sos>'] + words
            input = input[:self.max_sequence_length]

            target = words[:self.max_sequence_length-1]
            target = target + ['<eos>']

            assert len(input) == len(target), "%i, %i"%(len(input), len(target))
            length = len(input)

            input.extend(['<pad>'] * (self.max_sequence_length-length))
            target.extend(['<pad>'] * (self.max_sequence_length-length))

            input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
            target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

            id = len(data)
            data[id]['input'] = input
            data[id]['target'] = target
            data[id]['length'] = length

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self, song_entries):

        assert self.split == TRAIN_SET, "Vocabulary can only be created for training file."

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        for song_entry in song_entries:
            words = word_tokenize(song_entry.lyrics)
            w2c.update(words)

        for w, c in w2c.items():
            if c > self.min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocabulary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()
