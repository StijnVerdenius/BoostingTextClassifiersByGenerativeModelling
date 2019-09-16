import pandas as pd
import os
import numpy as np
import torch

def rewrite_labels(labels):
    """
    Rewrites each label class as a number.

    :param labels: The list of labels as type=str (w.r.t. entire dataset).
    :return: The list of labels as type=int.
    """

    new_labels = []

    # Answers: "Is this spam?"
    for label in labels:
        if label == "spam":
            new_labels.append(1)
        if label == "ham":
            new_labels.append(0)

    new_labels = np.asarray(new_labels)
    return new_labels

def fix_phrase_len(data, max_chars=50):
    new_data = list()
    for phrase in data:
        # check sentence length
        char_count = len(phrase)
        # truncate if sentence is too long
        if char_count > max_chars:
            new_phrase = phrase[:max_chars]
            new_data.append(new_phrase)
        # add padding if sentence is too short
        elif char_count < max_chars:
            padding_len = max_chars - char_count
            new_phrase = phrase + ' ' * padding_len
            new_data.append(new_phrase)
        else:
            new_data.append(phrase)

    new_data = np.asarray(new_data)
    return new_data

def create_vocab(data):
    # function goes through all data, and returns dict of unique_char:charID
    # will be used later to make one-hot vec encodings
    flattened = [d for sublist in data for d in sublist]
    unique_chars_list = sorted(set(flattened))
    vocabulary = dict((c, i) for i, c in enumerate(unique_chars_list))
    return vocabulary

def encode_data(data, vocabulary):
    encoded_data = list()

    for phrase in data:
        one_hot_vecs = list()

        # converting each phrase in our dataset into a list of chars
        chars = list(phrase)
        encoded_chars = [vocabulary[char] for char in chars]

        # create one hot vec of vocab len N
        # use charID to mark non-zero index
        for value in encoded_chars:
            letter = [0 for _ in range(len(vocabulary.keys()))]
            letter[value] = 1
            one_hot_vecs.append(np.asarray(letter))

        encoded_phrase = np.stack(one_hot_vecs, axis=1)
        encoded_data.append(encoded_phrase)

    # shape data set
    encoded_data = np.stack(encoded_data, axis=2)
    print(encoded_data.shape)
    return encoded_data



# hackey way to force Pycharm to be in intended root folder
os.chdir('../..')

# ================================
#          Get and read data
# ================================
df = pd.read_csv('local_data/data/spam_dataset.csv')
# outputs info about csv file
df.info()
# get info from pandas dataframe as np array
data = np.array(df.Message)
print('original data', data.shape)
labels = np.array(df.Category)

# ===================================
#          PRE-PROCESSING
# ===================================
# convert string labels into int labels
labels = rewrite_labels(labels)
# fix phrase length
max_chars = 50
data = fix_phrase_len(data, max_chars)

# one-hot encodings
# creating a dictionary of {unique char: charID}
vocabulary = create_vocab(data)
# return matrix of      unique_chars by max_chars by total_phrases_num
encoded_data = encode_data(data, vocabulary)


# convert to torch tensors
encoded_data = torch.from_numpy(encoded_data)
labels = torch.from_numpy(labels)

#================================
#          Sanity Checks
#================================
print('labels', labels.size())
print('data', encoded_data.size())














