import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from models.GeneralModel import GeneralModel

# this is a dummy, please make your own versions that inherets from general model and implements forward


class LSTMClassifier(GeneralModel):

    def __init__(self,
                 num_classes=10,
                 lstm_num_layers=2,
                 hidden_dim=256,
                 dropout=0.0,
                 batch_first=False,
                 n_channels_in=(0),
                 bidirectional=True,
                 device="cpu", **kwargs):

        super(LSTMClassifier, self).__init__(n_channels_in, device, **kwargs)

        self.lstm_num_layers = lstm_num_layers
        self.lstm_num_hidden = hidden_dim
        self.num_classes = num_classes
        self.device = device
        self.model = nn.LSTM(input_size=n_channels_in, hidden_size=hidden_dim,
                             num_layers=lstm_num_layers, dropout=dropout, batch_first=batch_first,
                             bidirectional=bidirectional)

        self.output_layer_toClass = nn.Linear(hidden_dim*2, num_classes, bias=True)

    def forward(self, x, lengths, h0=None, c0=None):
        batch_size, seq_length, vocab_size = x.shape  # todo implement check? # must turn 2(seq_len, batch, input_size)
        # x = x.permute([1, 0, -1]).to(self.device)
        x_packed = pack_padded_sequence(x, lengths, batch_first=True)
        if h0 is None:
            h0 = torch.zeros(self.lstm_num_layers*2, batch_size, self.lstm_num_hidden).to(self.device)
            c0 = torch.zeros(self.lstm_num_layers*2, batch_size, self.lstm_num_hidden).to(self.device)
        output, (h, c) = self.model(x_packed.float(), (h0.float(), c0.float()))
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.output_layer_toClass(output)
        return [output[-1, :, :]]  # , (h, c)
        # TODO concerns: take last, or average or sum ????
