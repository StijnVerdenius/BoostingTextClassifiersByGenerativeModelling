import os, torch, sys
from utils.system_utils import setup_directories, save_codebase_of_run
from utils.model_utils import calculate_accuracy
from torch.utils.data import DataLoader
from utils.constants import *
import pickle
from sklearn import metrics
from typing import List
from scipy.stats import ttest_ind
from statsmodels.stats.contingency_tables import mcnemar
from mlxtend.evaluate import permutation_test
import numpy as np

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

    def create_contingency_table(self, targets, predictions1, predictions2):
        assert len(targets) == len(predictions1)
        assert len(targets) == len(predictions2)

        contingency_table = np.zeros((2, 2))

        targets_length = len(targets)
        contingency_table[0, 0] = sum([targets[i] == predictions1[i] and targets[i] == predictions2[i] for i in range(targets_length)]) # both predictions are correct
        contingency_table[0, 1] = sum([targets[i] == predictions1[i] and targets[i] != predictions2[i] for i in range(targets_length)]) # predictions1 is correct and predictions2 is wrong
        contingency_table[1, 0] = sum([targets[i] != predictions1[i] and targets[i] == predictions2[i] for i in range(targets_length)]) # predictions1 is wrong and predictions2 is correct
        contingency_table[1, 1] = sum([targets[i] != predictions1[i] and targets[i] != predictions2[i] for i in range(targets_length)]) # both predictions are wrong

        return contingency_table

    def calculate_ttest(self, predictions1, predictions2):
        _, p_value = ttest_ind(predictions1, predictions2)
        return p_value

    def calculate_mcnemars_test(self, targets, predictions1, predictions2):
        contingency_table = self.create_contingency_table(
            targets,
            predictions1,
            predictions2)
        
        result = mcnemar(contingency_table, exact=True)
        return result.pvalue

    def calculate_permutation_test(self, predictions1, predictions2):
        p_value = permutation_test(predictions1, predictions2,
                            method='approximate',
                            num_rounds=10000,
                            seed=0)
                        
        return p_value

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


        targets = targets.detach().tolist()
        combined_predictions = combined_predictions.tolist()
        classifier_predictions = classifier_predictions.tolist()
        vaes_predictions = vaes_predictions.tolist()

        combined_f1, combined_precision, combined_recall = self.calculate_metrics(targets, combined_predictions)
        classifier_f1, classifier_precision, classifier_recall = self.calculate_metrics(targets, classifier_predictions)

        print(f'Combined F1: {combined_f1}\nClassifier F1: {classifier_f1}')

        mcnemars_p_value = self.calculate_mcnemars_test(targets, classifier_predictions, combined_predictions)
        ttest_p_value = self.calculate_ttest(classifier_predictions, combined_predictions)
        permutation_p_value = self.calculate_permutation_test(classifier_predictions, combined_predictions)

        alpha_value = 0.05
        mcnemars_significant = mcnemars_p_value < alpha_value
        ttest_significant = ttest_p_value < alpha_value
        permutation_significant = permutation_p_value < alpha_value

        print(f'Mcnemars: {mcnemars_significant} | p-value: {mcnemars_p_value}')
        print(f'T-Test: {ttest_significant} | p-value: {ttest_p_value}')
        print(f'Permutation: {permutation_significant} | p-value: {permutation_p_value}')

        # check if combination correctly classified these? check how many
        # print(combined_compare[classifier_misfire_indices])

        # print(classifier_misfire_indices)



