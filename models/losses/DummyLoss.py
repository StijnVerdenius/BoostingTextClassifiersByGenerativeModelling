import torch.nn as nn

from models.GeneralModel import GeneralModel

# this is a dummy, please make your own versions that inherets from general model and implements forward

class DummyLoss(GeneralModel):

    def __init__(self, device="cpu", **kwargs):
        super(DummyLoss, self).__init__(0, device, **kwargs)

    def forward(self, inp):
        pass
