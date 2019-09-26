import torch.nn as nn
import torch
from utils.constants import *
from utils.model_utils import find_right_model
from models.GeneralModel import GeneralModel
from models.enums import Genre

class VAEClassifier(GeneralModel):

    def __init__(self,
                 hidden_dim, z_dim,
                 vae_files, vaes_names,
                 generator_loss,
                 n_channels_in=(0),
                 device="cpu",
                 # if we dont write alternatives, we'll never need to change these two
                 generator_class='BaseVAE', only_eval=True,
                 **kwargs):

        super(VAEClassifier, self).__init__(n_channels_in, device, **kwargs)

        vae_files = vae_files.split(',')
        vaes_names = vaes_names.split(',')
        self.device = device

        self.loss_func = find_right_model(LOSS_DIR, generator_loss).to(device)
        self.models = [None]*len(vae_files)
        for v, vae_file in enumerate(vae_files):
            vae_file = os.path.join(GITIGNORED_DIR, RESULTS_DIR, vae_file)

            # index = Genre.Pop if 'Pop' in vae_file else (
            #     Genre.HipHop if 'Hip-Hop' in vae_file else (
            #         Genre.Rock if 'Rock' in vae_file else (
            #             Genre.Metal if 'Metal' in vae_file else (
            #                 Genre.Country if 'Country' in vae_file else(
            #                     )))))
            index = 0 if 'pop' in vae_file.lower() else (
                1 if 'hip-hop' in vae_file.lower() else (
                    2 if 'rock' in vae_file.lower() else (
                        3 if 'metal' in vae_file.lower() else (
                            4 if 'country' in vae_file.lower() else(
                                )))))

            # if 'rock' in vae_file:
            #     z_dim = 32

            self.models[index] = find_right_model(GEN_DIR,
                                                  generator_class,
                                                  hidden_dim=hidden_dim,
                                                  z_dim=z_dim,
                                                  n_channels_in=n_channels_in,
                                                  input_dim=n_channels_in,
                                                  device=device
                                                  ).to(device)

            # if 'rock' in vae_file:
            #     z_dim = 16

            if only_eval:
                self.models[index].eval()
            datamanager = DataManager(vae_file)
            print(vae_file)
            loaded = datamanager.load_python_obj(os.path.join('models', vaes_names[v]))
            for state_dict in loaded.values():
                state_dict = state_dict
            self.models[index].load_state_dict(state_dict)

        # print('same objects??')
        # for model in self.models:
        #     print(model is self.models[0])

    def forward(self, inp):
        regs, recons, = [], []  # todo make sure one forward doesn't affect the other? or the loss?
        for model in self.models:
            output = model.forward(inp.detach(), None)
            reg, recon = self.loss_func.test(None, *output)
            regs.append(reg)
            recons.append(recon)
        return regs, recons

