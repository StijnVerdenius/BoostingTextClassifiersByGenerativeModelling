import torch.nn as nn
import torch
from utils.constants import *
from utils.model_utils import find_right_model
from models.GeneralModel import GeneralModel


class VAEClassifier(GeneralModel):

    def __init__(self,
                 input_dim, hidden_dim, z_dim,
                 vae_files='',
                 n_channels_in=(0),
                 device="cpu",
                 loss='ELBO',  # if we dont write alternatives, we'll never need to change these two
                 generator_class='BaseVAE',
                 **kwargs):

        super(VAEClassifier, self).__init__(n_channels_in, device, **kwargs)
        vae_files = vae_files.split(',')
        self.loss_func = find_right_model(LOSS_DIR, loss).to(device)
        self.models = []
        for vae_file in vae_files:
            vae_file = os.path.join(GITIGNORED_DIR, RESULTS_DIR, vae_file)
            self.models.append(find_right_model(GEN_DIR,
                                                generator_class,
                                                hidden_dim=hidden_dim,
                                                z_dim=z_dim,
                                                n_channels_in=input_dim,
                                                input_dim=input_dim,
                                                device=device
                                                ).to(device))
            datamanager = DataManager(vae_file)
            loaded = datamanager.load_python_obj(os.path.join('models', 'KILLED_at_epoch_73'))
            for state_dict in loaded.values():
                state_dict = state_dict
            self.models[-1].load_state_dict(state_dict)
            self.device = device

    def forward(self, inp):
        regs, recons, = [], []  # todo make sure one forward doesn't affect the other? or the loss?
        for model in self.models:
            output = model.forward(inp.detach(), None)
            reg, recon = self.loss_func.test(None, *output)
            regs.append(reg)
            recons.append(recon)
        return regs, recons

