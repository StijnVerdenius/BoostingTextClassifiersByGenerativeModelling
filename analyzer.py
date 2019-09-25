import os, torch, sys
from utils.system_utils import setup_directories, save_codebase_of_run
from utils.model_utils import calculate_accuracy
from torch.utils.data import DataLoader
from utils.constants import *


class Analyzer:
    # input: both network models
    # return average loss, acc; etc.

    def __init__(self,
                 model, tester, num_classes,
                 model_state_path='',
                 device='cpu'):

        self.model = model  # be combined classifiers!!!
        self.tester = tester
        self.model_state_path = model_state_path  # todo hmmm
        self.device = device
        self.num_classes = num_classes
        self.model.eval()

    def analyze_misclassifications(self, test_logs):

        combined_scores = torch.stack(test_logs['final_scores']).view(-1, 5)
        classifier_scores = torch.stack(test_logs['combination']['classifier_scores']).view(-1, 5)
        vaes_scores = torch.stack(test_logs['combination']['vaes_scores']).view(-1, 5)
        targets = torch.stack(test_logs['true_targets']).view(-1).to(self.device)

        _, combined_predictions = combined_scores.max(dim=-1)
        _, classifier_predictions = classifier_scores.max(dim=-1)
        _, vaes_predictions = vaes_scores.max(dim=-1)

        # print('targets', targets)
        # print('combine', combined_predictions)
        # print('classif', classifier_predictions)
        # print('vaescla', vaes_predictions)

        classifier_compare = classifier_predictions.eq(targets)
        combined_compare = combined_predictions.eq(targets)
        vaes_compare = vaes_predictions.eq(targets)

        # print('Elbo values', vaes_scores)

        print('Accuracies:'
              '\n-Combined:', combined_compare.float().mean().item(),
              '\n-Base Classifier:', classifier_compare.float().mean().item(),
              '\n-Classify By Elbo:', vaes_compare.float().mean().item())

        classifier_misfire_indices = (classifier_compare == 0).nonzero()  # get misclassifications

        # check if combination correctly classified these? check how many
        # print(combined_compare[classifier_misfire_indices])

        # print(classifier_misfire_indices)



