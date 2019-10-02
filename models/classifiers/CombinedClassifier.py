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
                 generator_class,
                 dataset_options,
                 classifier_name, vaes_names,
                 # below: probably things we wont change
                 classifier_class='LSTMClassifier',
                 test_mode=False,
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
                                               only_eval=only_eval,
                                               dataset_options=dataset_options,
                                               test_mode=test_mode
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

        if self.combination_method == 'learn_classifier':

            self.layers_lstm = nn.Sequential(
                nn.Linear(5, 5),
                nn.ReLU()
            )

            self.layers_vae = nn.Sequential(
                nn.Linear(5, 5),
                nn.ReLU()
            )

            self.shared = nn.Sequential(
                nn.Linear(10, 5),
                nn.Sigmoid()
            )

        elif self.combination_method == "learn_sum":

            self.W_classifier = nn.Parameter(torch.ones((num_classes,))*0.5)
            self.W_vaes = nn.Parameter(torch.ones((num_classes,) )* 0.5)
            self.W_classifier.requires_grad = True
            self.W_vaes.requires_grad = True

    def forward(self, inp, targets, lengths, inp_sentence, step):
        inp2, targets2, lengths2 = inp_sentence
        out_base = self.base_classifier.forward(inp.detach(), lengths.detach())  # need to put through softmax
        out_base_class_scores = nn.functional.softmax(*out_base, dim=-1).detach()

        _, _, vae_loss = self.vae_classifier.forward(inp2, lengths2, step, targets2)
        vae_loss = torch.stack(vae_loss)
        vae_likelihood = - vae_loss
        vae_score = nn.Softmax(dim=-1)(vae_likelihood)

        if 'joint' in self.combination_method:
            combined_score = self.joint_probability(out_base_class_scores, vae_score)
        elif 'learn' in self.combination_method :
            combined_score = self.weighted_sum(out_base_class_scores, vae_score)

        # print(step, vae_score.tolist(), vae_likelihood.tolist(), targets.item(), combined_score.tolist())

        return combined_score, (out_base_class_scores, vae_likelihood)

        # todo | keep in mind that scores should be as if they're probabilities, meaning for
        # todo | classification we take argmax and classify directly

    def joint_probability(self, pxy, px):
        approach_px = torch.exp(px)
        return pxy*approach_px

    def weighted_sum(self, classifier_score, vaes_score):  # TODO: give elbo or recons or (regul??) ??
        if self.combination_method == "learn_classifier":
            return self.shared.forward(torch.cat((self.layers_lstm.forward(classifier_score.detach().squeeze()),
                                                  self.layers_vae.forward(vaes_score.detach())), dim=0))

        elif self.combination_method == "learn_sum":
            return self.W_classifier * classifier_score.detach() + self.W_vaes * vaes_score.detach()
