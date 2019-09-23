import torch.nn as nn
import torch
from utils.constants import *
from utils.model_utils import find_right_model
from models.GeneralModel import GeneralModel


class VAEClassifier(GeneralModel):

    def __init__(self, input_dim, hidden_dim, z_dim,
                 vae_files=[],
                 generator_class='BaseVAE',
                 n_channels_in=(0),
                 device="cpu",
                 **kwargs):
        super(VAEClassifier, self).__init__(n_channels_in, device, **kwargs)

        self.models = []
        for vae_file in vae_files:
            self.models.append(find_right_model(GEN_DIR,
                                                generator_class,
                                                hidden_dim=hidden_dim,
                                                z_dim=z_dim,
                                                input_dim=input_dim,
                                                device=DEVICE
                                                )).to(DEVICE)

    def forward(self, inp):
        pass
