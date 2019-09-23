import torch.nn as nn
import torch
from utils.constants import *
from utils.model_utils import find_right_model
from models.GeneralModel import GeneralModel


class VAEClassifier(GeneralModel):

    def __init__(self,
                 input_dim, hidden_dim, z_dim,
                 vae_files=[],
                 n_channels_in=(0),
                 device="cpu",
                 loss='ELBO',  # if we dont write alternatives, we'll never need to change these two
                 generator_class='BaseVAE',
                 **kwargs):

        super(VAEClassifier, self).__init__(n_channels_in, device, **kwargs)

        self.loss_func = find_right_model(LOSS_DIR, loss).to(device)
        self.models = []
        for vae_file in vae_files:
            self.models.append(find_right_model(GEN_DIR,
                                                generator_class,
                                                hidden_dim=hidden_dim,
                                                z_dim=z_dim,
                                                input_dim=input_dim,
                                                device=device
                                                )).to(device)

            self.models[-1].load_state_dict(torch.load(vae_file))

    def forward(self, inp):
        losses = []  # todo make sure one forward doesn't affect the other? or the loss?
        for model in self.models:
            output = model.forward(inp.detach())
            losses.append(self.loss_func.forward(_, *output))
        return losses
