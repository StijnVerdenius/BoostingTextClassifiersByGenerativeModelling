import torch.nn as nn
import torch

from models.GeneralModel import GeneralModel

# this is a dummy, please make your own versions that inherets from general model and implements forward

class DummyGenerator(GeneralModel):

    def __init__(self, n_channels_in=(0), device="cpu", **kwargs):
        super(DummyGenerator, self).__init__(n_channels_in, device, **kwargs)

    def forward(self, inp):
        pass

    def parameters(self): #don't implement self.parameters() in your own versions
        return [torch.zeros((2,2))]