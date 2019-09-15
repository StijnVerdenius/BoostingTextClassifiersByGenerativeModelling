from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# hard coded filenames etc because its sole purpose it to check if model works


class CheckDataLoader(Dataset):

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
