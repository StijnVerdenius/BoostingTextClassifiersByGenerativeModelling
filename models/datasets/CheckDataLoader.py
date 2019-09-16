from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

# hard coded filenames etc because its sole purpose it to check if model works


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


class CheckDataLoader(Dataset):

    def __init__(self, file="", set_name="train"):
        super(CheckDataLoader, self).__init__()
        df = pd.read_csv('local_data/spam_dataset.csv')
        data = np.array(df.Message)
        labels = np.array(df.Category)

        # ========= Preprocess data ===========
        # convert string labels into int labels
        labels = rewrite_labels(labels)
        # fix phrase length
        max_chars = 50
        data = fix_phrase_len(data, max_chars)

        # one-hot encodings
        # creating a dictionary of {unique char: charID}
        self.vocabulary = create_vocab(data)

        # return matrix of      unique_chars by max_chars by total_phrases_num
        encoded_data = encode_data(data, self.vocabulary)

        # convert to torch tensors
        self.encoded_data = torch.from_numpy(encoded_data.astype(float))  # V x Seq x B
        self.labels = torch.from_numpy(labels)
        self.encoded_data = self.encoded_data.permute([-1, 1, 0])  # B x Seq x Vocab

    def __len__(self):
        return self.encoded_data.shape[0]

    def __getitem__(self, item):
        # item im hoping is an integer to index
        return self.encoded_data[item, :, :], self.labels[item].long()
        # vocab size x sequence length

