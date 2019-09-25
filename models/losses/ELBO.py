import torch
import torch.distributions as dd
import torch.nn as nn

from models.GeneralModel import GeneralModel


######## inspired by; https://github.com/kefirski/contiguous-succotash

class ELBO(GeneralModel):

    def __init__(self, device="cpu", **kwargs):
        super(ELBO, self).__init__(0, device, **kwargs)
        self.losses = {"recon": 0, "reg": 0}
        self.evaluations = 0

        self.normal = None

    def forward(self, _, mean, std, reconstructed_mean, x):
        """
        calculates negated-ELBO loss
        """

        self.evaluations += 1
        x = x.float()

        # regularisation loss
        loss_reg = (- 0.5 * torch.sum(1 + torch.log(std ** 2) - mean ** 2 - std ** 2, dim=-1)).mean()

        # reconstruction loss
        # try:
        loss_recon = -1*dd.Normal(reconstructed_mean, 1).log_prob(x).mean()
        # loss_recon = dd.MultivariateNormal(reconstructed_mean, scale_tril=1).log_prob(x)
        # except Exception as e:
        #     if self.normal is None:
        #         self.normal = .to(self.device)
        #     else:
        #         raise e
        #     loss_recon = self.normal.log_prob(x)

        # nn.MSELoss(reduction="mean")(reconstructed_mean, x) #* (98/100)
        # loss_recon += nn.L1Loss(reduction="mean")(reconstructed_mean, x)/100
        # loss_recon += nn.CosineSimilarity()(reconstructed_mean, x).mean()/1000

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
        loss_reg = (- 0.5 * torch.sum(1 + torch.log(std ** 2) - mean ** 2 - std ** 2, dim=-1))
        loss_recon = nn.MSELoss(reduction='none')(reconstructed_mean, x)

        loss_recon = loss_recon.mean(-1)
        loss_reg = loss_reg.mean(-1)

        return loss_reg.detach(), loss_recon.detach()  # / batch_size
