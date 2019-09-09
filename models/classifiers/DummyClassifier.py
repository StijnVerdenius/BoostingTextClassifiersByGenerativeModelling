import torch.nn as nn
import torch

from models.GeneralModel import GeneralModel


class DummyClassifier(GeneralModel):

    def __init__(self, n_channels_in=(0), device="cpu", **kwargs):
        super(DummyClassifier, self).__init__(n_channels_in, device, **kwargs)

    def forward(self, inp):
        pass

    def parameters(self):
        return [torch.zeros((2,2))]