import torch.nn as nn
import torch
from utils.constants import *
from utils.model_utils import find_right_model
from models.GeneralModel import GeneralModel
from models.classifiers import VAEClassifier


class CombinedClassifier(GeneralModel):

    def __init__(self,
                 input_dim, hidden_dim, z_dim,
                 num_classes,
                 lstm_file,
                 vae_files,
                 classifier_name, vaes_names,
                 # below: probably things we wont change
                 classifier_class='LSTMClassifier',
                 generator_class='BaseVAE',
                 generator_loss='ELBO',
                 n_channels_in=(0), device="cpu", **kwargs):
        super(CombinedClassifier, self).__init__(n_channels_in, device, **kwargs)

        self.base_classifier = find_right_model(CLASS_DIR,
                                                classifier_class,
                                                n_channels_in=n_channels_in,
                                                num_classes=num_classes,
                                                device=device,
                                                hidden_dim=hidden_dim
                                                ).to(device)

        self.vae_classifier = find_right_model(CLASS_DIR,
                                               'VAEClassifier',
                                               generator_class=generator_class,
                                               generator_loss=generator_loss,
                                               vae_files=vae_files,
                                               vaes_names=vaes_names,
                                               device=device,
                                               input_dim=input_dim,
                                               hidden_dim=hidden_dim, z_dim=z_dim,
                                               ).to(device)

        lstm_file = os.path.join(GITIGNORED_DIR, RESULTS_DIR, lstm_file)
        datamanager = DataManager(lstm_file)
        loaded = datamanager.load_python_obj(os.path.join('models', classifier_name))
        for state_dict in loaded.values():
            state_dict = state_dict
        self.base_classifier.load_state_dict(state_dict)

    def forward(self, inp, lengths):
        out_base = self.base_classifier.forward(inp.detach(), lengths.detach())  # need to put through softmax
        out_base_class_scores = torch.nn.functional.softmax(out_base[0], dim=-1).detach()

        out_vaes_regul, out_vaes_reconst = self.vae_classifier.forward(inp)

        # expected: both tensors to be values per class so: B x C
        out_vaes_regul = torch.stack(out_vaes_regul).permute([-1, 0])
        out_vaes_reconst = torch.stack(out_vaes_reconst).permute([-1, 0])

        # todo combine scores somehow?
        # lowest loss is the best don't forget. todo, take inv or neg?
        combined_score = self.joint_probability(out_base_class_scores,
                                                out_vaes_regul,
                                                out_vaes_reconst)

        print('lstm', out_base_class_scores)
        print('vreg', out_vaes_regul)
        print('vrec', out_vaes_reconst)
        print('comb', combined_score)

        return combined_score, (out_base_class_scores, out_vaes_regul, out_vaes_reconst)

        # todo keep in mind that scores should be as if they're probabilities, meaning for
        # classification we take argmax and classify directly

    def joint_probability(self, pxy, regul, recon):
        elbo = - regul+recon
        approach_px = torch.exp(elbo)
        return pxy*approach_px
