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
                                                device=device
                                                ).to(device)

        self.base_classifier.load_state_dict(torch.load(lstm_file))

        self.vae_classifier = find_right_model(CLASS_DIR,
                                               'VAEClassifier',
                                               generator_class=generator_class,
                                               generator_loss=generator_loss,
                                               vae_files=vae_files,
                                               device=device,
                                               input_dim=input_dim,
                                               hidden_dim=hidden_dim, z_dim=z_dim,
                                               ).to(device)

    def forward(self, inp):
        out_base = self.base_classifier.forward(inp.detach())  # need to put through softmax
        out_base_class_scores = out_base.log_softmax(dim=-1)

        out_vaes_loss_per_class = self.vae_classifier.forward(inp)

        # expected: both tensors to be values per class so: B x C
        print(out_base_class_scores.shape, out_vaes_loss_per_class.shape)
        # combine scores somehow?
        # lowest loss is the best don't forget. todo, take inv or neg?

        combined_score = out_base_class_scores  # todo: what to do? check how scores are
        return combined_score, out_base_class_scores, out_vaes_loss_per_class

