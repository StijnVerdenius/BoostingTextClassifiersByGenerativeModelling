from models.losses.ELBO import *


class MSE_ELBO(ELBO):

    def __init__(self, device="cpu", **kwargs):
        super(MSE_ELBO, self).__init__(device=device, **kwargs)

    def get_recon_loss(self, x, recon):
        return nn.MSELoss(reduction="none")(recon, x)
