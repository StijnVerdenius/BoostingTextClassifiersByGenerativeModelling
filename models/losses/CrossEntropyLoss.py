import torch.nn as nn

from models.GeneralModel import GeneralModel


class CrossEntropyLoss(GeneralModel):

    def __init__(self, n_channels_in=(0), device="cpu", **kwargs):
        super(CrossEntropyLoss, self).__init__(n_channels_in, device, **kwargs)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inp, target):
        return self.loss(inp, target)
