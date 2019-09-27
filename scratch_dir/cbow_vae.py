import numpy as np
import torch
import pandas as pd

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


def create_vocab(data):
    # function goes through all data, and returns dict of unique_char:charID
    # will be used later to make one-hot vec encodings
    # flattened = [d for sublist in data for d in sublist]
    unique_chars_list = sorted(set(data))
    vocabulary = dict((c, i) for i, c in enumerate(unique_chars_list))
    return vocabulary

def encode_data(bag_of_words, vocabulary):
    one_hot_vecs = list()

    for word in bag_of_words:
        word_ID = vocabulary[word]
        encoded_word = [0 for _ in range(len(vocabulary.keys()))]
        encoded_word[word_ID] = 1
        one_hot_vecs.append(np.asarray(encoded_word))

    one_hot_vecs = np.stack(one_hot_vecs, axis=1)
    one_hot_vecs = torch.from_numpy(one_hot_vecs)
    print(one_hot_vecs.shape)
    # encoded_data.append(encoded_phrase)
    #
    # # shape data set
    # encoded_data = np.stack(encoded_data, axis=2)
    # print(encoded_data.shape)
    # return encoded_data

df = pd.read_csv('/Users/bellanicholson/Desktop/DL4NLP/code/DL4NLP/local_data/data/spam_dataset.csv')
# todo: remove [:80], just done for testing purposes
data = np.array(df.Message)[:80]
labels = np.array(df.Category)[:80]

# ========== MAIN =============
# convert string labels into int labels
labels = rewrite_labels(labels)
#
data = ",".join(data).strip().split(' ')
vocabulary = create_vocab(data)
encoded_data = encode_data(data, vocabulary)