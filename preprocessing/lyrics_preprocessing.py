import os
import csv
import numpy as np
import sys
import re

sys.path.append('..')

from utils.data_manager import DataManager
from utils.system_utils import ensure_current_directory
from models.entities.Song import Song
from models.enums.Genre import Genre

def save_dataset_text(song_entries, embeddings_folder_path, filename):
    embeddings_filename = os.path.join(embeddings_folder_path, filename)
    with open(embeddings_filename, 'w', encoding='utf8') as embeddings_file:
        # for _, song_entries in dataset.items():
        for song_entry in song_entries:
            embeddings_file.write(f'{song_entry.lyrics}\n')


ensure_current_directory()
main_path = os.path.join('local_data', 'data')

dataset_folder_path = os.path.join(main_path, '380000-lyrics-from-metrolyrics')
if not os.path.exists(dataset_folder_path):
    raise Exception('Dataset folder does not exist')

dataset_file_path = os.path.join(dataset_folder_path, 'lyrics.csv')
if not os.path.exists(dataset_file_path):
    raise Exception('Dataset file does not exist')

with open(dataset_file_path, 'r', encoding="utf8") as dataset_file:
    next(dataset_file)  # skip the first line - headers
    dataset_reader = csv.reader(dataset_file, delimiter=',')

    classes_to_skip = ['Other', 'Not Available']

    song_entries_by_genre = {}

    lines_counter = 0

    for _, row in enumerate(dataset_reader):
        if row[4] in classes_to_skip:
            continue

        # Cut off songs which have lyrics with less than 30 characters
        if len(row[5]) < 100:
            continue
        
        if not any(c.isalpha() for c in row[5]):
            continue

        song_genre = Genre.from_str(row[4])
        
        # if the song genre is None, it means it's not supported, so we skip this song
        if not song_genre:
            continue

        song_entry = Song(song_genre, row[5])

        if row[4] not in song_entries_by_genre.keys():
            song_entries_by_genre[row[4]] = [song_entry]
        else:
            song_entries_by_genre[row[4]].append(song_entry)

genres = list(song_entries_by_genre.keys())
songs_limit = 13000
for genre in genres:
    song_entries_by_genre[genre] = song_entries_by_genre[genre][:songs_limit]

train_data, validation_data, test_data = {}, {}, {}

embeddings_folder_path = os.path.join(main_path, 'embeddings')
if not os.path.exists(embeddings_folder_path):
    os.mkdir(embeddings_folder_path)

for genre, song_entries in song_entries_by_genre.items():
    # split data into train, validation and test sets
    train_data[genre] = song_entries[:int(len(song_entries)*0.7)]
    validation_data[genre] = song_entries[int(len(song_entries)*0.7):int(len(song_entries)*0.8)]
    test_data[genre] = song_entries[int(len(song_entries)*0.8):]

    # # print statistics
    min_length = min([len(song_entry.lyrics) for song_entry in song_entries])
    max_length = max([len(song_entry.lyrics) for song_entry in song_entries])
    avg_length = np.mean([len(song_entry.lyrics) for song_entry in song_entries])
    print(f'{genre} - min: {min_length} | max: {max_length} | mean: {avg_length} | all: {len(song_entries)} | train: {len(train_data[genre])} | validation: {len(validation_data[genre])} | test: {len(test_data[genre])}')

train_song_entries = sorted([item for value in train_data.values() for item in value], key=lambda song: len(song.lyrics))
validation_song_entries = sorted([item for value in validation_data.values() for item in value], key=lambda song: len(song.lyrics))
test_song_entries = sorted([item for value in test_data.values() for item in value], key=lambda song: len(song.lyrics))

empties_train, empties_test, empties_val = [],[],[]
for i, song in enumerate(train_song_entries):
    if song.lyrics == '':
        empties_train.append(i)
for i, song in enumerate(test_song_entries):
    if song.lyrics == '':
        empties_test.append(i)
for i, song in enumerate(validation_song_entries):
    if song.lyrics == '':
        empties_val.append(i)

for index in sorted(empties_train, reverse=True):
    del train_song_entries[index]
indexes = [2, 3, 5]
for index in sorted(empties_test, reverse=True):
    del test_song_entries[index]
indexes = [2, 3, 5]
for index in sorted(empties_val, reverse=True):
    del validation_song_entries[index]

lines_counter = 0
for song in train_song_entries:
    song.start_index = lines_counter
    lines_counter += song.number_of_lines

lines_counter = 0
for song in validation_song_entries:
    song.start_index = lines_counter
    lines_counter += song.number_of_lines
    
lines_counter = 0
for song in test_song_entries:
    song.start_index = lines_counter
    lines_counter += song.number_of_lines

data_manager = DataManager(main_path)
data_manager.save_python_obj(train_song_entries, 'song_lyrics.train')
data_manager.save_python_obj(validation_song_entries, 'song_lyrics.validation')
data_manager.save_python_obj(test_song_entries, 'song_lyrics.test')

save_dataset_text(train_song_entries, embeddings_folder_path, 'embeddings.train.txt')
save_dataset_text(validation_song_entries, embeddings_folder_path, 'embeddings.validation.txt')
save_dataset_text(test_song_entries, embeddings_folder_path, 'embeddings.test.txt')