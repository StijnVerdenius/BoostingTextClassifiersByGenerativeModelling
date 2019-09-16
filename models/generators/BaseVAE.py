import torch
import torch.nn as nn

from models.GeneralModel import GeneralModel
from models.losses.ELBO import ELBO


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

    def __init__(self, n_in, hidden_dim, z_dim, device="cpu"):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=z_dim, out_channels=hidden_dim, kernel_size=1, dilation=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=n_in, kernel_size=1, dilation=2),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        mean = self.layers(x)
        return mean.permute(0, 2, 1)


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
        epsilon: torch.Tensor

        # reperimatrization-sample z
        z = epsilon.__mul__(std).__add__(mean)

        # recosntruct x from z
        self.decoder: Decoder
        reconstruction_mean = self.decoder.forward(z)

        return mean, std, reconstruction_mean.contiguous().view((batch_size, -1)), x.contiguous().view((batch_size, -1))


if __name__ == '__main__':
    testbatch = torch.randn((128, 20, 100))

    vae = BaseVAE(n_channels_in=100)
    vae: BaseVAE
    x = vae.forward(testbatch)
    lossfunc = ELBO()
    score = lossfunc.forward(*x)
    print(score.shape, score)
