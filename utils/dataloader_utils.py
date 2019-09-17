import numpy as np

import torch

def pad_and_sort_batch(DataLoaderBatch):
    """
    DataLoaderBatch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest, 
    """
    batch_size = len(DataLoaderBatch)
    batch_split = list(zip(*DataLoaderBatch))

    sequences, targets = batch_split[0], batch_split[1]
    
    lengths = [len(sequence) for sequence in sequences]
    max_length = max(lengths)

    embedding_dimension = sequences[0].shape[1]

    padded_sequences = np.ones((batch_size, max_length, embedding_dimension))

    for i, l in enumerate(lengths):
        padded_sequences[i][0:l][:] = sequences[i][0:l][:]
    
    return _sort_batch(torch.from_numpy(padded_sequences), torch.tensor(targets), torch.tensor(lengths))

def _sort_batch(batch, targets, lengths):
    """
    Sort a minibatch by the length of the sequences with the longest sequences first
    return the sorted batch targes and sequence lengths.
    This way the output can be used by pack_padded_sequences(...)
    """
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    target_tensor = targets[perm_idx]
    return seq_tensor, target_tensor, seq_lengths