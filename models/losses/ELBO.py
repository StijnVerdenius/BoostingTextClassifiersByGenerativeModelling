import torch
import torch.distributions as dd
import torch.nn as nn

from models.GeneralModel import GeneralModel


######## inspired by; https://github.com/kefirski/contiguous-succotash

class ELBO(GeneralModel):

    def __init__(self, device="cpu", **kwargs):
        self.losses = {"recon": 0, "reg": 0}
        self.evaluations = 0
        self.normal = None
        super(ELBO, self).__init__(0, device=device, **kwargs)

    def forward(self, _, mean, std, reconstructed_mean, x):
        """
        calculates negated-ELBO loss
        """

        self.evaluations += 1
        x = x.float()

        # regularisation loss
        loss_reg = self.get_reg_loss(std, mean).mean()
        loss_recon = self.get_recon_loss(x, reconstructed_mean).mean()

        self.losses["recon"] += loss_recon.item()
        self.losses["reg"] += loss_reg.item()

        # average over batch size
        return loss_recon + loss_reg  # / batch_size

    def get_recon_loss(self, x, recon):
        raise NotImplementedError("overrided by childclass! please choose one")

    def get_losses(self):
        return {key: value / self.evaluations for key, value in self.losses.items()}

    def reset(self):
        self.losses["recon"] = 0
        self.losses["reg"] = 0
        self.evaluations = 0

    def get_reg_loss(self, std, mean):
        return - 0.5 * torch.sum(1 + torch.log(std ** 2) - mean ** 2 - std ** 2, dim=-1)

    def test(self, _, mean, std, reconstructed_mean, x):
        """
        calculates negated-ELBO loss
        """

        self.evaluations += 1

        x = x.float()
        loss_reg = self.get_reg_loss(std, mean)
        loss_recon = self.get_recon_loss(x, reconstructed_mean) #nn.MSELoss(reduction='none')(reconstructed_mean, x)

        loss_recon = loss_recon.mean(-1)
        loss_reg = loss_reg.mean(-1)

        return loss_reg.detach(), loss_recon.detach()  # / batch_size
