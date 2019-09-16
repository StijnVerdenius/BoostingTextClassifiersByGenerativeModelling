from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# hard coded filenames etc because its sole purpose it to check if model works


class CheckDataLoader(Dataset):

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

        new_data = np.asarray(new_data)
        return new_data

    def create_vocab(data):
        # function goes through all data, and returns dict of unique_char:charID
        # will be used later to make one-hot vec encodings
        flattened = [d for sublist in data for d in sublist]
        unique_chars_list = sorted(set(flattened))
        vocabulary = dict((c, i) for i, c in enumerate(unique_chars_list))
        return vocabulary

    def encode_data(data, vocabulary, max_chars=50):
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
        return encoded_data

    def __init__(self, file="", set_name="train"):
        super(CheckDataLoader, self).__init__()
        print('Initializing CheckDataLoader- model checker!')
        df = pd.read_csv('local_data/spam_dataset.csv')
        data = np.array(df.Message)
        labels = np.array(df.Category)
        max_chars = 100

        # ========= Preprocess data ===========
        new_data = []
        for sentence in data:
            # check sentence length
            char_count = len(sentence)
            # truncate if sentence is too long
            if char_count > max_chars:
                new_sentence = sentence[:max_chars]
                new_data.append(new_sentence.lower())
            # add padding if sentence is too short
            elif char_count < max_chars:
                padding_len = max_chars - char_count
                new_sentence = sentence.ljust(padding_len)
                new_data.append(new_sentence.lower())
        self.data = np.asarray(new_data)

        # ======= Preprocess Labels ===============
        # Answers: Is the text spam?
        new_labels = []
        for label in labels:
            if label == "spam":
                new_labels.append(1)
            if label == "ham":
                new_labels.append(0)
        self.labels = np.asarray(new_labels)

        # get unique vocab
        flattened = [d for sublist in new_data for d in sublist]
        vocabulary = sorted(set(flattened))
        print('Number of unique characters (vocabulary):', len(vocabulary))
        print(vocabulary)

        #TODO convert self.data to one-hot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # item im hoping is an integer to index
        #TODO this is not complete,
        #TODO it will index the list then convert it to torch tensor
        return {'data': self.data[item], 'label': self.labels[item]}  # not sure if this works
