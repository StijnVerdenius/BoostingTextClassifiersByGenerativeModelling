import torch.nn as nn
from models.losses import CrossEntropyLoss, MSE_ELBO
from models.GeneralModel import GeneralModel

# this is a dummy, please make your own versions that inherets from general model and implements forward

class Cross_ELBO(GeneralModel):

    def __init__(self, device="cpu", **kwargs):
        super(Cross_ELBO, self).__init__(0, device, **kwargs)
        self.CE = CrossEntropyLoss()
        self.MSE_ELBO = MSE_ELBO()

    def forward(self, y, x, recons, yhat):
        return self.MSE_ELBO(recons, x) + self.CE(yhat, y)

