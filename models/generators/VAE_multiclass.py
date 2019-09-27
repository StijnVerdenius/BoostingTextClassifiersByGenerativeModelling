import torch.nn as nn
import torch

from models.GeneralModel import GeneralModel
import numpy as np

# this is a dummy, please make your own versions that inherets from general model and implements forward

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, z_dim, num_classes=2, device="cpu"):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device

        # self.activation = nn.ReLU().to(device)
        self.num_classes = num_classes
        input_dim += self.num_classes  # because of label concat
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0.1, bias=True, num_layers=2)

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(device)

        self.y_to_hidden = nn.Linear(1, hidden_dim).to(device)
        self.layer_mu = nn.Linear(hidden_dim, z_dim).to(device)
        self.layer_logvar = nn.Linear(hidden_dim, z_dim).to(device)

    def forward(self, x, labels):
        print(x.shape)
        B, S, _ = x.shape
        x = x.float()

        labels_onehot = torch.zeros((B, self.num_classes)).to(self.device)
        labels_onehot[np.arange(0, B), labels] = 1
        labels_onehot_repeated = labels_onehot.unsqueeze(1).repeat(1, S, 1)
        xy = torch.cat((x, labels_onehot_repeated), dim=-1)
        print(xy.shape)
        lstm_output, _ = self.lstm.forward(xy)
        shared = self.layers.forward(lstm_output)

        mean = self.layer_mu(shared)
        logvar = self.layer_logvar(shared)  # deleted activation
        std = torch.sqrt(torch.exp(logvar))
        return mean, std

class Decoder(nn.Module):

    def __init__(self, n_in, hidden_dim, z_dim, device="cpu"):
        super(Decoder, self).__init__()

        self.decoder_rnn = nn.RNN(n_in, hidden_dim, num_layers=z_dim, bidirectional=self.bidirectional,
                                  batch_first=True)

    def forward(self, x_in):
        self.decoder_rnn(x_in)


class VAE_multiclass(GeneralModel):

    def __init__(self, hidden_dim, z_dim,
                 n_channels_in=(0), device="cpu", **kwargs):
        super(VAE_multiclass, self).__init__(n_channels_in, device, **kwargs)
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device
        self.encoder = Encoder(n_channels_in, hidden_dim, z_dim, device=device)

    def forward(self, x, lengths=None, step=None, labe=None):
        x = x.to(self.device)
        label = label.to(self.device)

        self.encoder: Encoder
        mean, std = self.encoder.forward(x, label)

        x, recons_x, yhat = 0, 0, 0
        return x, recons_x, yhat

    def sample(self):
        z = torch.randn((1, 25, self.z_dim))

        x = self.decoder.forward(z).squeeze()

        x: torch.Tensor

        distr = torch.nn.Softmax(1)(x)

        # y = distr.argmax(dim=1)
        y = torch.multinomial(distr, 20)

        return y

    def compare_metric(self, current_best, loss, accuracy) -> bool:
        if current_best[0] > loss:
            return True

        return False