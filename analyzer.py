import os, torch, sys
from utils.system_utils import setup_directories, save_codebase_of_run
from utils.model_utils import calculate_accuracy
from torch.utils.data import DataLoader
from utils.constants import *
import pickle

class Analyzer:
    # input: both network models
    # return average loss, acc; etc.

    def __init__(self,
                 model, num_classes,
                 model_state_path='',
                 device='cpu'):

        self.model = model  # be combined classifiers!!!
        self.model_state_path = model_state_path  # todo hmmm
        self.device = device
        self.num_classes = num_classes
        self.model.eval()

    def soft_voting(self, probs1, probs2):
        _, predictions = ((probs1 + probs2) / 2).max(dim=-1)
        return predictions

    def analyze_misclassifications(self, test_logs):
        if test_logs is not None:
            with open('logs1k.pickle', 'wb') as handle:
                pickle.dump(test_logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('logs1k.pickle', 'rb') as handle:
                test_logs = pickle.load(handle)

        combined_scores = torch.stack(test_logs['final_scores']).view(-1, 5)
        classifier_scores = torch.stack(test_logs['combination']['classifier_scores']).view(-1, 5)
        vaes_scores = torch.stack(test_logs['combination']['vaes_scores']).view(-1, 5)
        targets = torch.stack(test_logs['true_targets']).view(-1).to(self.device)

        _, combined_predictions = combined_scores.max(dim=-1)
        _, classifier_predictions = classifier_scores.max(dim=-1)
        _, vaes_predictions = vaes_scores.max(dim=-1)

        # combined_predictions = self.soft_voting(vaes_scores, classifier_scores)
        # print('targets', targets)
        # print('combine', combined_predictions)
        # print('classif', classifier_predictions)
        # print('vaescla', vaes_predictions)

        classifier_compare = classifier_predictions.eq(targets)
        combined_compare = combined_predictions.eq(targets)
        vaes_compare = vaes_predictions.eq(targets)

        classifier_misfire_indices = (classifier_compare == 0).nonzero()  # get misclassifications
        vae_improved = vaes_compare[classifier_misfire_indices].float().mean()
        print('VAE classified', vae_improved, 'of the LSTM misclassifications correctly.')

        # print('Elbo values', vaes_scores)

        print('Accuracies:'
              '\n-Combined:', combined_compare.float().mean().item(),
              '\n-Base Classifier:', classifier_compare.float().mean().item(),
              '\n-Classify By Elbo:', vaes_compare.float().mean().item())


        # check if combination correctly classified these? check how many
        # print(combined_compare[classifier_misfire_indices])

        # print(classifier_misfire_indices)



