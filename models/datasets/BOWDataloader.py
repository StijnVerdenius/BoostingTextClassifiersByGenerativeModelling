from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from collections import Counter
import nltk
nltk.download('punkt')

class BOWDataloader(Dataset):

    def __init__(self, file="", set_name="train", **kwargs):
        super(BOWDataloader, self).__init__()

        df = pd.read_csv('/Users/bellanicholson/Desktop/DL4NLP/code/DL4NLP/local_data/data/spam_dataset.csv')
        # todo: remove [:80], just done for testing purposes
        raw_data = np.array(df.Message)[:80]
        labels = np.array(df.Category)[:80]

        # convert string labels into int labels
        self.labels = self.rewrite_labels(labels)

        # tokenize all word in raw data
        words_list = self.tokenize_words(raw_data)
        # all the unique words in the data set
        unique_words = set(words_list)

        vocabulary = self.create_vocab(unique_words)
        vocabulary, self.normalizing_vec = self.calc_word_freq(words_list, vocabulary)
        self.encoded_data = self.encode_data(raw_data, vocabulary)

        # converting things into tensors
        self.labels = torch.from_numpy(self.labels)
        self.encoded_data = torch.from_numpy(self.encoded_data)
        self.normalizing_vec = torch.from_numpy(self.normalizing_vec)


    def tokenize_words(self, raw_text):

        processed_text = []
        # tokenizing words by line
        for line in raw_text:
            processed_line = nltk.word_tokenize(line.lower())
            processed_text.extend(processed_line)
        return processed_text

    def rewrite_labels(self, labels):
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


    # edited so it words with words and not chars
    def create_vocab(self, data):
        # function goes through all data, and returns dict of unique_char:charID
        # will be used later to make one-hot vec encodings
        # flattened = [d for sublist in data for d in sublist]
        unique_chars_list = sorted(data)
        vocabulary = dict((c, i) for i, c in enumerate(unique_chars_list))
        return vocabulary

    def calc_word_freq(self, dataset, vocabulary):
        """
        Calculate word frequency in our dataset.

        :param dataset: Appends word_count to vocab
        :param vocabulary: {"word": wordID}
        :return: vocabulary --> {"word": [wordID, word_count]}
        """
        normalizing_vec = []
        word_counts = Counter(dataset)
        for word in vocabulary.keys():
            word_ID = vocabulary[word]
            vocabulary[word] = [word_ID, word_counts[word]]
            normalizing_vec.append(word_counts[word])
        return vocabulary, np.asarray(normalizing_vec)


    # TODO: check this function
    def encode_data(self, raw_data, vocabulary):
        """
        Uses vocabulary to build a one hot vec to correspond to each word. Appends

        :param words_list: Unprocessed dataset
        :param vocabulary: {"word": word_ID, word_freq}
        :return: vocabulary as
        """
        # encode by line
        encoded_data = []
        # loop through text line by line
        for line in raw_data:
            # print('LINE', line)
            # processing text
            words = nltk.word_tokenize(line.lower())
            # creating empty tensor that will show word counts of all words in that particular line
            encoded_line = [0 for _ in range(len(vocabulary.keys()))]
            for word in words:
                wordID, word_freq = vocabulary[word]
                encoded_line[wordID] = word_freq
            encoded_data.append(encoded_line)
        encoded_data = np.stack(encoded_data, axis=0)
        return encoded_data


    def __len__(self):
        return self.encoded_data.shape[0]

    def __getitem__(self, item):
        # item im hoping is an integer to index
        return self.encoded_data[item, :], self.labels[item].long()
        # return self.encoded_data[item, :] / self.norm_vec, self.labels[item].long()

# # Dummy test
# x = DataLoader(CheckDataLoader())
# for i, tens in enumerate(x):
#     print(i)


