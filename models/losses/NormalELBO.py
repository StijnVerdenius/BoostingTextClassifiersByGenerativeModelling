from models.losses.ELBO import *


class NormalELBO(ELBO):

    def __init__(self, device="cpu", **kwargs):
        super(NormalELBO, self).__init__(device=device, **kwargs)

    def get_recon_loss(self, x, recon):
        return -1*dd.Normal(recon, 1).log_prob(x)