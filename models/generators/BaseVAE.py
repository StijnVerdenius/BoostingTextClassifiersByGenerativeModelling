import torch
import torch.nn as nn
import torch.nn.functional as pytorch_functions

from models.GeneralModel import GeneralModel


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, z_dim, device="cpu"):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.activation = nn.ReLU().to(device)

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0.1, bias=True)

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(device)

        self.layer_mu = nn.Linear(hidden_dim, z_dim).to(device)
        self.layer_sig = nn.Linear(hidden_dim, z_dim).to(device)

    def forward(self, x):
        lstm_output = self.lstm.forward(x)[0]
        shared = self.layers.forward(lstm_output)

        mean = self.layer_mu(shared)
        std = self.activation(self.layer_sig(shared))

        return mean, std

import os





class Decoder(nn.Module):

    def __init__(self, n_in, hidden_dim, z_dim, device="cpu"):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.layers = nn.Sequential(
            # nn.Conv1d(...),
            # nn.Softmax(),
            # nn.Conv1d(...),
            # nn.Softmax(),
        ).to(device)

    def forward(self, x):
        mean = self.layers(x)
        return mean


class BaseVAE(GeneralModel):

    def __init__(self, n_channels_in=(0), hidden_dim=100, z_dim=20, device="cpu", **kwargs):
        super(BaseVAE, self).__init__(n_channels_in, device, **kwargs)

        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device
        self.encoder = Encoder(n_channels_in, hidden_dim, z_dim, device=device)
        self.decoder = Decoder(n_channels_in, hidden_dim, z_dim, device=device)

    def forward(self, x: torch.Tensor):  # todo: revisit
        # ensure device
        x = x.to(self.device)

        # get Q(z|x)
        self.encoder: Encoder
        mean, std = self.encoder.forward(x)

        # obtain batch size
        batch_size = x.shape[0]

        # sample an epsilon
        epsilon = torch.randn(mean.shape).to(self.device)

        # reperimatrization-sample z
        z = epsilon.mul(std).add(mean)

        # recosntruct x from z
        self.decoder: Decoder
        reconstruction_mean = self.decoder.forward(z)

        # get loss
        elbo = self.elbo(mean, std, reconstruction_mean, x.view((batch_size, -1)))

        x.detach()

        return elbo

    @staticmethod
    def elbo(mean, std, reconstructed_mean, x):  # todo: revisit
        """
        calculates negated-ELBO loss
        :param mean:
        :param std:
        :param reconstructed_mean:
        :param x:
        :return:
        """

        # get batch size
        batch_size = x.shape[0]

        # regularisation loss
        loss_reg = torch.sum(torch.sum(-1 * torch.log(std + 1e-7) + ((std.pow(2) + mean.pow(2)) - 1) * 0.5, dim=1),
                             dim=0)
        # reconstruction loss
        loss_recon = pytorch_functions.cross_entropy(reconstructed_mean, x, reduction="sum")

        # average over batch size
        return (loss_recon + loss_reg) / batch_size


if __name__ == '__main__':
    testbatch = torch.randn((128, 20, 100))

    vae = BaseVAE(n_channels_in=100)
    vae: BaseVAE
    x = vae.encoder.forward(testbatch)
    # print(x.shape)
    print(os.pathsep)

