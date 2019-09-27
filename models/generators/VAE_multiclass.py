import torch.nn as nn
import torch

from models.GeneralModel import GeneralModel
import numpy as np
from torch.autograd import Variable

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, z_dim, num_classes, device="cpu"):
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

    def forward(self, x, lengths, labels):
        B, S, _ = x.shape
        x = x.float()
        labels_onehot = torch.zeros((B, self.num_classes)).to(self.device)
        labels_onehot[np.arange(0, B), labels] = 1
        labels_onehot_repeated = labels_onehot.unsqueeze(1).repeat(1, S, 1)
        xy = torch.cat((x, labels_onehot_repeated), dim=-1)

        xy_packed = pack_padded_sequence(xy, lengths, batch_first=True)
        lstm_output, _ = self.lstm.forward(xy_packed)
        lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)

        shared = self.layers.forward(lstm_output)

        mean = self.layer_mu(shared)
        logvar = self.layer_logvar(shared)  # deleted activation
        std = torch.sqrt(torch.exp(logvar))
        return mean, std

class Decoder(nn.Module):

    def __init__(self, n_in, hidden_dim, out_dim, num_classes, device="cpu"):
        super(Decoder, self).__init__()

        self.decoder_rnn = nn.RNN(n_in, hidden_dim, num_layers=2, bidirectional=True,
                                  batch_first=True)

        self.linear_x = nn.Linear(hidden_dim*2, out_dim)
        self.linear_y = nn.Linear(hidden_dim*2, num_classes)


    def forward(self, x_in, lengths):
        x_packed = pack_padded_sequence(x_in, lengths, batch_first=True)
        out, _ = self.decoder_rnn(x_packed)
        out, _ = pad_packed_sequence(out, batch_first=True)

        out_x = self.linear_x(out)
        out_y = self.linear_y(out).sum(dim=1)

        return out_x, out_y


class VAE_multiclass(GeneralModel):

    def __init__(self, hidden_dim, z_dim, num_classes,
                 n_channels_in=(0), device="cpu", **kwargs):

        super(VAE_multiclass, self).__init__(n_channels_in, device, **kwargs)
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device
        self.encoder = Encoder(n_channels_in, hidden_dim, z_dim, num_classes, device=device)
        self.decoder = Decoder(z_dim, hidden_dim, n_channels_in, num_classes=num_classes, device=device)


    def forward(self, x, lengths=None, step=None, label=None):
        x = x.to(self.device)
        label = label.to(self.device)

        self.encoder: Encoder
        mean, std = self.encoder.forward(x, lengths, label)
        z = self.sample_z(mean, std)
        recons_x, yhat = self.decoder.forward(z, lengths)
        return x, recons_x, yhat, mean, std

    def sample_z(self, mean, std):
        noise = Variable(torch.randn(mean.size())).to(self.device)
        return (noise * std) + mean

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