from utils.constants import *
from utils.model_utils import find_right_model
from models.GeneralModel import GeneralModel
from models.enums.Genre import Genre

class VAEClassifier(GeneralModel):

    def __init__(self,
                 hidden_dim, z_dim,
                 vae_files, vaes_names,
                 generator_loss, generator_class,
                 dataset_options,
                 n_channels_in=(0),
                 device="cpu",
                 only_eval=True,
                 **kwargs):

        super(VAEClassifier, self).__init__(n_channels_in, device, **kwargs)

        vae_files = vae_files.split(',')
        vaes_names = vaes_names.split(',')
        self.device = device

        self.loss_func = find_right_model(LOSS_DIR, generator_loss, dataset_options=dataset_options).to(device)

        self.MSEELBOOOOOO = find_right_model(LOSS_DIR, 'MSE_ELBO', dataset_options=dataset_options).to(device)

        self.models = [None]*len(vae_files)
        for v, vae_file in enumerate(vae_files):
            vae_file = os.path.join(GITIGNORED_DIR, RESULTS_DIR, vae_file)

            index = Genre.Pop.value if 'pop' in vae_file.lower() else (
                Genre.HipHop.value if 'hip-hop' in vae_file.lower() else (
                    Genre.Rock.value if 'rock' in vae_file.lower() else (
                        Genre.Metal.value if 'metal' in vae_file.lower() else (
                            Genre.Country.value if 'country' in vae_file.lower() else(
                                )))))
            self.models[index] = find_right_model(GEN_DIR,
                                                  generator_class,
                                                  hidden_dim=hidden_dim,
                                                  z_dim=z_dim,
                                                  n_channels_in=n_channels_in,
                                                  input_dim=n_channels_in,
                                                  device=device,
                                                  embedding_size=n_channels_in,
                                                  latent_size=z_dim,
                                                  dataset_options=dataset_options
                                                  ).to(device)

            if only_eval:
                self.models[index].eval()
            datamanager = DataManager(vae_file)
            loaded = datamanager.load_python_obj(os.path.join('models', vaes_names[v]))
            for state_dict in loaded.values():
                state_dict = state_dict
            self.models[index].load_state_dict(state_dict)

        # print('same objects??')
        # for model in self.models:
        #     print(model is self.models[0])

    def forward(self, inp, lengths, step, targets):
        regs, recons, losses = [], [], []  # todo make sure one forward doesn't affect the other? or the loss?
        LOSS = 'TOD'

        for m, model in enumerate(self.models):
            output = model.forward(inp.detach(), lengths, step)

            if LOSS == 'STIJN':
                _, _, _, x0, batch_size, logp, mean, logv, z, std = output
                print(logp.shape, inp.shape)
                print(inp)
                reg, recon = self.MSEELBOOOOOO.test(None, mean, std, logp, inp)
                regs.append(reg)
                recons.append(recon)
            else:
                loss = self.loss_func.forward(targets, *output)
                losses.append(loss)

        return regs, recons, losses

