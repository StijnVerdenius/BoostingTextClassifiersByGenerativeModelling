import pandas as pd
import os
import numpy as np
from sklearn import preprocessing

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
    new_data = []
    for sentence in data:
        # check sentence length
        char_count = len(sentence)
        # truncate if sentence is too long
        if char_count > max_chars:
            new_sentence = sentence[:max_chars]
            new_data.append(new_sentence)
        # add padding if sentence is too short
        elif char_count < max_chars:
            padding_len = max_chars - char_count
            new_sentence = len(sentence) + padding_len
            new_data.append(new_sentence)
    new_data = np.asarray(new_data)
    return new_data

def create_vocab(data):
    flattened = [d for sublist in data for d in sublist]
    vocabulary = sorted(set(flattened))
    char_to_int = dict((c, i) for i, c in enumerate(vocabulary))
    return vocabulary, char_to_int
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
labels = np.array(df.Category)

# ===================================
#          PRE-PROCESSING
# ===================================
# convert string labels into int labels
labels = rewrite_labels(labels)
# fix phrase length
max_chars = 100
data = fix_phrase_len(data, max_chars)

# one-hot encodings
# creating a dictionary of all unique chars in our vocab
vocabulary, char_to_int = create_vocab(data)

# flattened = [d for sublist in data for d in sublist]
# vocabulary = sorted(set(flattened))
# char_to_int = dict((c, i) for i, c in enumerate(vocabulary))


# TODO: clean up code
encoded_data = list()
for phrase in data:
    one_hot_vecs = list()
    # converting each phrase in our dataset into a list of chars
    words = phrase.split()
    for word in words:
        chars = list(word)
        # creating a list of charIDs (using char_to_int dictionary)
        encoded_chars = [char_to_int[char] for char in chars]
        # create one hot vec of vocab len N
        # use charID to mark non-zero index
        for value in encoded_chars:
            letter = [0 for _ in range(len(vocabulary))]
            letter[value] = 1
            one_hot_vecs.append(letter)
    encoded_data.append(one_hot_vecs)










