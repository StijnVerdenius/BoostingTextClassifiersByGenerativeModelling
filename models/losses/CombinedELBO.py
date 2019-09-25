from models.losses.ELBO import *


class CombinedELBO(ELBO):

    def __init__(self, device="cpu", **kwargs):
        super(CombinedELBO, self).__init__(device=device, **kwargs)

    def get_recon_loss(self, x, recon):
        return -1*dd.Normal(recon, 1).log_prob(x) + nn.MSELoss(reduction="none")(recon, x)