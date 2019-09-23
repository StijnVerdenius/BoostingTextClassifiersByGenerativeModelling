import torch
import torch.nn as nn

from models.GeneralModel import GeneralModel


class ELBO(GeneralModel):

    def __init__(self, device="cpu", **kwargs):
        super(ELBO, self).__init__(0, device, **kwargs)
        self.losses = {"recon": 0, "reg": 0}
        self.evaluations = 0

    def forward(self, _, mean, std, reconstructed_mean, x):  # todo: revisit
        """
        calculates negated-ELBO loss
        """

        self.evaluations += 1
        # todo: revisit
        x = x.float()

        # regularisation loss
        loss_reg = (-0.5 * torch.sum(std - torch.pow(mean, 2) - torch.exp(std) + 1, 1)).mean().squeeze()  #
        # loss_reg = -0.5 * torch.sum(std - mean.pow(2) - std.exp() + 1, 1).mean()

        # loss_reg = torch.sum(torch.sum(-1 * torch.log(std + 1e-7) + ((std.pow(2) + mean.pow(2)) - 1) * 0.5, dim=1),
        #                      dim=0)

        # reconstruction loss
        loss_recon = nn.MSELoss(reduction="mean")(reconstructed_mean, x)

        self.losses["recon"] += loss_recon.item()
        self.losses["reg"] += loss_reg.item()

        # average over batch size
        return loss_recon + loss_reg  # / batch_size

    def get_losses(self):
        return {key: value / self.evaluations for key, value in self.losses.items()}

    def reset(self):
        self.losses["recon"] = 0
        self.losses["reg"] = 0
        self.evaluations = 0

    def test(self, _, mean, std, reconstructed_mean, x):  # todo: revisit
        """
        calculates negated-ELBO loss
        """

        self.evaluations += 1

        x = x.float()
        loss_reg = (-0.5 * torch.sum(std - torch.pow(mean, 2) - torch.exp(std) + 1, 1))
        loss_recon = nn.MSELoss(reduction='none')(reconstructed_mean, x)

        loss_recon = loss_recon.mean(-1)
        loss_reg = loss_reg.mean(-1)

        return loss_reg.detach(), loss_recon.detach()  # / batch_size