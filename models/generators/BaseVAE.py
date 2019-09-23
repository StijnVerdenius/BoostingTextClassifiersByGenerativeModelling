import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import numpy as np

import random

from models.GeneralModel import GeneralModel
from models.datasets.CheckDataLoader import CheckDataLoader
from models.losses.ELBO import ELBO
from utils.constants import SEED, DEVICE
from utils.data_manager import DataManager
from utils.system_utils import ensure_current_directory


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, z_dim, device="cpu"):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.activation = nn.ReLU().to(device)

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0.1, bias=True, num_layers=2)

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(device)

        self.layer_mu = nn.Linear(hidden_dim, z_dim).to(device)
        self.layer_sig = nn.Linear(hidden_dim, z_dim).to(device)

    def forward(self, x):
        x = x.float()
        lstm_output = self.lstm.forward(x)[0]
        shared = self.layers.forward(lstm_output)

        mean = self.layer_mu(shared)
        std = self.activation(self.layer_sig(shared))

        return mean, std


class Decoder(nn.Module):

    def effective_k(self, k, d):
        return (k - 1) * d + 1

    def __init__(self, n_in, hidden_dim, z_dim, device="cpu"):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.decoder_dilations = [1, 2, 4]
        self.decoder_kernels = [(400, self.z_dim, 3),
                                (450, 400, 3),
                                (n_in, 450, 3)]
        self.decoder_paddings = [self.effective_k(w, self.decoder_dilations[i]) - 1
                                 for i, (_, _, w) in enumerate(self.decoder_kernels)]

        self.kernels = [Parameter(torch.Tensor(out_chan, in_chan, width).normal_(0, 0.05))
                        for out_chan, in_chan, width in self.decoder_kernels]
        self._add_to_parameters(self.kernels, 'decoder_kernel')

        self.biases = [Parameter(torch.Tensor(out_chan).normal_(0, 0.05))
                       for out_chan, in_chan, width in self.decoder_kernels]
        self._add_to_parameters(self.biases, 'decoder_bias')

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=z_dim, out_channels=hidden_dim, kernel_size=1, dilation=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=n_in, kernel_size=1, dilation=2),
            nn.Sigmoid(),
        ).to(device)

    def _add_to_parameters(self, parameters, name):
        for i, parameter in enumerate(parameters):
            self.register_parameter(name='{}-{}'.format(name, i), param=parameter)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer, kernel in enumerate(self.kernels):
            # apply conv layer with non-linearity and drop last elements of sequence to perfrom input shifting
            x = F.conv1d(x, kernel,
                         bias=self.biases[layer],
                         dilation=self.decoder_dilations[layer],
                         padding=self.decoder_paddings[layer])

            x_width = x.size()[2]
            x = x[:, :, :(x_width - self.decoder_paddings[layer])].contiguous()

            x = F.relu(x)
        mean = x.permute(0, 2, 1)
        return mean


class BaseVAE(GeneralModel):

    def __init__(self, n_channels_in=(0), hidden_dim=100, z_dim=20, device="cpu", **kwargs):
        super(BaseVAE, self).__init__(n_channels_in, device, **kwargs)

        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device
        self.encoder = Encoder(n_channels_in, hidden_dim, z_dim, device=device)
        self.decoder = Decoder(n_channels_in, hidden_dim, z_dim, device=device)

    def forward(self, x: torch.Tensor, _):  # todo: revisit
        # ensure device
        x = x.to(self.device)

        # get Q(z|x)
        self.encoder: Encoder
        mean, std = self.encoder.forward(x)

        # obtain batch size
        batch_size = x.shape[0]

        # sample an epsilon
        epsilon = torch.randn(mean.shape).to(self.device)
        epsilon: torch.Tensor

        # reperimatrization-sample z
        z = epsilon.__mul__(std).__add__(mean)

        # recosntruct x from z
        self.decoder: Decoder
        reconstruction_mean = self.decoder.forward(z)

        return mean, std, reconstruction_mean.contiguous().view((batch_size, -1)), x.contiguous().view((batch_size, -1))

    def sample(self):

        z = torch.randn((self.z_dim, 20, self.hidden_dim))

        x = self.decoder.forward(z)

        x: torch.Tensor

        y = x.argmax(dim=2)

        return y





def _test_sample_vae():
    vae = BaseVAE(n_channels_in=106, hidden_dim=128, z_dim=128)
    vae: BaseVAE

    datamanager = DataManager("./local_data/results/spamham")

    loaded = datamanager.load_python_obj("models/KILLED_at_epoch_2")

    state_dict = 0
    for state_dict in loaded.values():
        state_dict = state_dict

    vae.load_state_dict(state_dict)

    y = vae.sample()

    data = CheckDataLoader()

    vocab = data.vocabulary

    vocab_rev = {value: key for key, value in vocab.items()}

    for sen in y:
        string = ""
        for num in sen:
            string += (vocab_rev[num.item()])

        print(string)


def _test_vae_forward():
    testbatch = torch.randn((128, 20, 100))  # batch, seq_len, embedding

    vae = BaseVAE(n_channels_in=100, hidden_dim=128, z_dim=128)
    vae: BaseVAE

    x = tuple([None]) + vae.forward(testbatch, None)
    lossfunc = ELBO()
    score = lossfunc.forward(*x)
    print(score.shape, score)




if __name__ == '__main__':
    ensure_current_directory()
    if DEVICE == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(SEED)

    # for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    _test_sample_vae()
    _test_vae_forward()

