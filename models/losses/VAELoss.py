import torch
import torch.nn as nn
import numpy as np

from models.GeneralModel import GeneralModel

class VAELoss(GeneralModel):

    def __init__(self, dataset_options, device="cpu", test_mode=False,
    **kwargs):
        super(VAELoss, self).__init__(0, device, **kwargs)
        self.losses = {"recon": 0, "reg": 0}
        self.evaluations = 0
        self.test_mode = test_mode
        self.NLL = torch.nn.NLLLoss(size_average=False, ignore_index=dataset_options.pad_idx)

    def forward(self, target, length, step, k, x0, batch_size, logp, mean, logv, z, _):
        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))
        
        # Negative Log Likelihood
        NLL_loss = self.NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = self.kl_anneal_function('logistic', step, k, x0)

        # return NLL_loss, KL_loss, KL_weight
        loss = (NLL_loss + KL_weight * KL_loss)/batch_size
        return loss

    def get_losses(self):
        return {key: value / 1 for key, value in self.losses.items()}

    def kl_anneal_function(self, anneal_function, step, k, x0, test_mode=True):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)


    def reset(self):
        self.losses["recon"] = 0
        self.losses["reg"] = 0
        self.evaluations = 0