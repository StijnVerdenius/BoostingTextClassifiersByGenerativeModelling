import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from models.GeneralModel import GeneralModel

# this is a dummy, please make your own versions that inherets from general model and implements forward


class LSTMClassifier(GeneralModel):

    def __init__(self,
                 num_classes=10,
                 lstm_num_layers=1,
                 hidden_dim=256,
                 dropout=0.0,
                 batch_first=True,
                 embedding_size=256,
                 bidirectional=True,
                 device="cpu", **kwargs):

        super(LSTMClassifier, self).__init__(embedding_size, device, **kwargs)

        self.lstm_num_layers = lstm_num_layers
        self.lstm_num_hidden = hidden_dim
        self.num_classes = num_classes
        self.device = device
        self.model = nn.LSTM(input_size=embedding_size, hidden_size=hidden_dim,
                             num_layers=lstm_num_layers, dropout=dropout, batch_first=batch_first,
                             bidirectional=bidirectional)

        self.output_layer_toClass = nn.Linear(hidden_dim*2, num_classes, bias=False)

    def forward(self, x, lengths, **kwargs):

        x_packed = pack_padded_sequence(x, lengths, batch_first=True)

        output, (_, _) = self.model(x_packed.float())
        output, _ = pad_packed_sequence(output, batch_first=True)

        output = self.output_layer_toClass(output)
        return [output.sum(dim=1)]

    def compare_metric(self, current_best, loss, accuracy) -> bool:
        if current_best[1] < accuracy:
            return True

        return False
