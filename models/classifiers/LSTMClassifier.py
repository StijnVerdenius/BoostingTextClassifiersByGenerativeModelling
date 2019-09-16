import torch.nn as nn
import torch

from models.GeneralModel import GeneralModel

# this is a dummy, please make your own versions that inherets from general model and implements forward


class LSTMClassifier(GeneralModel):

    def __init__(self,
                 vocabulary_size,
                 num_classes,
                 lstm_num_layers=2,
                 lstm_num_hidden=256,
                 dropout=0.0,
                 batch_first=False,
                 n_channels_in=(0),
                 device="cpu", **kwargs):

        super(LSTMClassifier, self).__init__(n_channels_in, device, **kwargs)

        self.lstm_num_layers = lstm_num_layers
        self.lstm_num_hidden = lstm_num_hidden
        self.num_classes = num_classes
        self.device = device
        self.model = nn.LSTM(input_size=vocabulary_size, hidden_size=lstm_num_hidden,
                             num_layers=lstm_num_layers, dropout=dropout, batch_first=batch_first)

        self.output_layer = nn.Linear(lstm_num_hidden, num_classes,
                                      bias=True)

    def forward(self, x, h0=None, c0=None):
        batch_size, seq_length, vocab_size = x.shape  # todo implement check? # must turn 2(seq_len, batch, input_size)
        x = x.permute([1, 0, -1]).to(self.device)
        if h0 is None:
            h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_num_hidden).to(self.device)
            c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_num_hidden).to(self.device)
        output, (h, c) = self.model(x.float(), (h0.float(), c0.float()))
        output = self.output_layer(output)
        return [output]  # , (h, c)
