import torch.nn as nn
import torch
from utils.constants import *
from utils.model_utils import find_right_model
from models.GeneralModel import GeneralModel
from models.classifiers import VAEClassifier
from torch.autograd import Variable


class CombinedClassifier(GeneralModel):

    def __init__(self,
                 hidden_dim, z_dim,
                 hidden_dim_vae, num_classes,
                 lstm_file, vae_files,
                 generator_loss,
                 classifier_name, vaes_names,
                 # below: probably things we wont change
                 classifier_class='LSTMClassifier',
                 generator_class='BaseVAE',
                 combination_method='joint',
                 only_eval=True,
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
                                               n_channels_in=n_channels_in,
                                               hidden_dim=hidden_dim_vae,
                                               z_dim=z_dim,
                                               only_eval=only_eval
                                               ).to(device)

        if only_eval:
            self.base_classifier.eval()

        lstm_file = os.path.join(GITIGNORED_DIR, RESULTS_DIR, lstm_file)
        datamanager = DataManager(lstm_file)
        loaded = datamanager.load_python_obj(os.path.join('models', classifier_name))

        for state_dict in loaded.values():
            state_dict = state_dict
        self.base_classifier.load_state_dict(state_dict)

        # in case we wanna learn a weighted sum
        # TODO I guess this should be saved n loaded as well?
        self.combination_method = combination_method

        if self.combination_method is 'learn':
            self.W_classifier = nn.Parameter(torch.ones((num_classes,)*0.5))
            self.W_vaes = nn.Parameter(torch.ones((num_classes,) * 0.5))
            self.W_classifier.requires_grad = True
            self.W_vaes.requires_grad = True

    def forward(self, inp, lengths):
        out_base = self.base_classifier.forward(inp.detach(), lengths.detach())  # need to put through softmax
        out_base_class_scores = nn.functional.softmax(out_base[0], dim=-1).detach()

        out_vaes_regul, out_vaes_reconst = self.vae_classifier.forward(inp)

        # expected: both tensors to be values per class so: B x C
        out_vaes_regul = torch.stack(out_vaes_regul).permute([-1, 0])
        out_vaes_reconst = torch.stack(out_vaes_reconst).permute([-1, 0])
        out_elbo = - out_vaes_regul + out_vaes_reconst  # (- negative elbo)

        if self.combination_method is 'joint':
            combined_score = self.joint_probability(out_base_class_scores,
                                                    out_vaes_regul,
                                                    out_vaes_reconst)
        elif self.combination_method is 'learn':
            combined_score = self.weighted_sum(out_base_class_scores, out_elbo)

        # print('lstm', out_base_class_scores)
        # print('vreg', out_vaes_regul)
        # print('vrec', out_vaes_reconst)
        # print('comb', combined_score)

        return combined_score, (out_base_class_scores, out_elbo)

        # todo | keep in mind that scores should be as if they're probabilities, meaning for
        # todo | classification we take argmax and classify directly

    def joint_probability(self, pxy, regul, recon):

        elbo = regul+recon
        approach_px = torch.exp(-elbo)

        print('Regularization', regul)
        print('Reconstruction', recon)
        print('ELBO', elbo)
        print('exp elbo', approach_px)
        # print(approach_px)
        # print(pxy)
        # print(pxy*approach_px)

        return pxy*approach_px

    def weighted_sum(self, classifier_score, vaes_score):  # TODO: give elbo or recons or (regul??) ??
        return self.W_classifier * classifier_score + self.W_vaes * vaes_score
