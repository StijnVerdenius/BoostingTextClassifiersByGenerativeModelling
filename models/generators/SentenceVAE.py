import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils

from models.GeneralModel import GeneralModel
# from models.datasets.CheckDataLoader import CheckDataLoader
from models.datasets.LyricsRawDataset import LyricsRawDataset
from models.losses.ELBO import ELBO
from models.enums.Genre import Genre
from utils.data_manager import DataManager
from utils.system_utils import ensure_current_directory
from utils.constants import *

class SentenceVAE(GeneralModel):

    def __init__(
        self,
        embedding_size,
        hidden_dim,
        latent_size,
        dataset_options,
        max_sequence_length=50,
        rnn_type='gru',
        embedding_dropout=0.5,
        word_dropout=0,
        num_layers=1,
        bidirectional=True,
        device="cpu",
        **kwargs):
        super(SentenceVAE, self).__init__(0, device, **kwargs)

        self.tensor = torch.cuda.FloatTensor #if torch.cuda.is_available() else torch.Tensor

        self.k = 0.0025 
        self.x0 = 2500

        self.max_sequence_length = max_sequence_length
        self.sos_idx = dataset_options.sos_idx
        self.eos_idx = dataset_options.eos_idx
        self.pad_idx = dataset_options.pad_idx
        self.unk_idx = dataset_options.unk_idx

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(dataset_options.vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_dim, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_dim, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_dim * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_dim * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_dim * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_dim * (2 if bidirectional else 1), dataset_options.vocab_size)

    def forward(self, input_sequence, lengths, step, **kwargs):

        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_dim*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = self.to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        # DECODER
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_dim)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            #if torch.cuda.is_available():
            prob=prob.cuda()
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b, s, _ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return lengths, step, self.k, self.x0, batch_size, logp, mean, logv, z, std

    def inference(self, n=4, z=None):

        # if z is None:
        #     batch_size = n
        #     z = to_var(torch.randn([batch_size, self.latent_size]))
        # else:
        batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_dim)

        # hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t=0
        while(t<self.max_sequence_length and len(running_seqs)>0):

            if t == 0:
                input_sequence = torch.Tensor(batch_size).fill_(self.sos_idx).cuda().long()

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z


    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to


    def sample(self):
        # z = torch.randn((1, 25, self.latent_size))
        z = torch.randn([2, self.latent_size]).cuda()

        # x = self.decoder_rnn.forward(z).squeeze()

        # x: torch.Tensor
        x, _ = self.inference(z=z)

        # distr = torch.nn.Softmax(1)(x.cpu())

        # # y = distr.argmax(dim=1)
        # y = torch.multinomial(distr, 20)

        return x

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def compare_metric(self, current_best, loss, accuracy) -> bool:
        if current_best[0] > loss:
            return True

        return False

    def to_var(self, x, volatile=False):
        # if torch.cuda.is_available():
        x = x.cuda()
        return Variable(x, volatile=volatile)

def _test_sample_vae():
    train_dataset = LyricsRawDataset(os.path.join('local_data', 'data'), TRAIN_SET, genre=Genre.Rock)

    vae = SentenceVAE(
        vocab_size=train_dataset.vocab_size,
        sos_idx=train_dataset.sos_idx,
        eos_idx=train_dataset.eos_idx,
        pad_idx=train_dataset.pad_idx,
        unk_idx=train_dataset.unk_idx,
        max_sequence_length=50,
        embedding_size=300,
        rnn_type='gru',
        hidden_dim=64,
        latent_size=32).cuda()
    # vae: SentenceVAE

    datamanager = DataManager("./local_data/results/2019-09-26_19.18.29")

    loaded = datamanager.load_python_obj("models/model_best")

    state_dict = 0
    for state_dict in loaded.values():
        state_dict = state_dict

    vae.load_state_dict(state_dict)

    vae.eval()

    y = vae.sample()

    for sen in y:
        string = ""
        for num in sen:
            string += (train_dataset.i2w[str(num.item())])

        print(string)

if __name__ == '__main__':
    seed = 42

    ensure_current_directory()
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

    # for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    _test_sample_vae()
    # _test_vae_forward()
    # _test_grouping_vae()
    # _test_reconstruction_vae()
