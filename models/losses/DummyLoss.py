import torch.nn as nn

from models.GeneralModel import GeneralModel

# this is a dummy, please make your own versions that inherets from general model and implements forward

class DummyLoss(GeneralModel):

    def __init__(self, n_channels_in=(0), device="cpu", **kwargs):
        super(DummyLoss, self).__init__(n_channels_in, device, **kwargs)

    def forward(self, inp):
        pass