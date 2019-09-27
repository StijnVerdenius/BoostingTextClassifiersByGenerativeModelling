import torch.nn as nn
from models.losses import CrossEntropyLoss, MSE_ELBO
from models.GeneralModel import GeneralModel
from utils.model_utils import find_right_model
from utils.constants import *
# this is a dummy, please make your own versions that inherets from general model and implements forward

class Cross_ELBO(GeneralModel):

    def __init__(self, dataset_options, device="cpu", **kwargs):
        super(Cross_ELBO, self).__init__(0, device, **kwargs)
        self.CE = find_right_model(LOSS_DIR, 'CrossEntropyLoss', dataset_options=dataset_options, device=device).to(device)

        self.MSE_ELBO = find_right_model(LOSS_DIR, 'MSE_ELBO', dataset_options=dataset_options, device=device).to(device)
        self.saved_losses = {'ce': None, 'mseelbo': None}

    def forward(self, y, x, recons, yhat, mean, std):
        mse_elbo = self.MSE_ELBO(None, mean, std, recons, x)
        ce = self.CE(y, yhat)
        self.saved_losses = {'ce': ce.item(), 'mseelbo': mse_elbo.item()}
        return mse_elbo + ce

    def reset(self):
        self.MSE_ELBO.reset()

    def get_losses(self):
        return self.saved_losses
