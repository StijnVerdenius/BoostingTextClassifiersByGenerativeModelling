import torch.nn as nn
import torch
from models.GeneralModel import GeneralModel


class CrossEntropyLoss(GeneralModel):

    def __init__(self, n_channels_in=(0), device="cpu", **kwargs):
        super(CrossEntropyLoss, self).__init__(n_channels_in, device, **kwargs)
        self.loss = nn.CrossEntropyLoss().to(device)

    def forward(self, target, out_data):
        return self.loss(out_data.to(self.device), target.to(self.device))
