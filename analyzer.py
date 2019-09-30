import os, torch, sys
from utils.system_utils import setup_directories, save_codebase_of_run
from utils.model_utils import calculate_accuracy
from torch.utils.data import DataLoader
from utils.constants import *
from sklearn import metrics
from typing import List

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

    def soft_voting(self, probs1, probs2):
        _, predictions = ((probs1 + probs2) / 2).max(dim=-1)
        return predictions
        
    def calculate_metrics(
        self,
        targets: List,
        predictions: List,
        average: str = "weighted"):

        if sum(predictions) == 0:
            return 0, 0, 0

        precision = metrics.precision_score(
            targets, predictions, average=average)
        recall = metrics.recall_score(targets, predictions, average=average)
        f1 = metrics.f1_score(targets, predictions, average=average)

        return f1, precision, recall


    def analyze_misclassifications(self, test_logs):

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


        # print('Elbo values', vaes_scores)

        print('Accuracies:'
              '\n-Combined:', combined_compare.float().mean().item(),
              '\n-Base Classifier:', classifier_compare.float().mean().item(),
              '\n-Classify By Elbo:', vaes_compare.float().mean().item())

        classifier_misfire_indices = (classifier_compare == 0).nonzero()  # get misclassifications

        targets = targets.detach().tolist()
        combined_predictions = combined_predictions.tolist()
        classifier_predictions = classifier_predictions.tolist()
        vaes_predictions = vaes_predictions.tolist()

        combined_f1, combined_precision, combined_recall = self.calculate_metrics(targets, combined_predictions)
        classifier_f1, classifier_precision, classifier_recall = self.calculate_metrics(targets, classifier_predictions)

        print(f'Combined F1: {combined_f1}\nClassifier F1: {classifier_f1}')

        # check if combination correctly classified these? check how many
        # print(combined_compare[classifier_misfire_indices])

        # print(classifier_misfire_indices)



