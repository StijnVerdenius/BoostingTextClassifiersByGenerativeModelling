import os, torch, sys
from utils.system_utils import setup_directories, save_codebase_of_run
from utils.model_utils import calculate_accuracy
from torch.utils.data import DataLoader
from utils.constants import *
import pickle
from sklearn import metrics
from typing import List
import numpy as np
from torch import nn
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
        print(probs1)
        return (probs1 + probs2) / 2
        
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

        if test_logs is not None:
            with open('logs_full_on_full.pickle', 'wb') as handle:
                pickle.dump(test_logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('logs_full_on_full.pickle', 'rb') as handle:
                test_logs = pickle.load(handle)

        combined_scores = torch.stack(test_logs['final_scores']).view(-1, 5).cpu()
        classifier_scores = torch.stack(test_logs['combination']['classifier_scores']).view(-1, 5).cpu()
        vaes_scores = torch.stack(test_logs['combination']['vaes_scores']).view(-1, 5).cpu()
        targets = torch.stack(test_logs['true_targets']).view(-1)

        # combined_scores = self.soft_voting(vaes_scores, classifier_scores)

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

        print('Accuracies:'
              '\n-Combined:', combined_compare.float().mean().item(),
              '\n-Base Classifier:', classifier_compare.float().mean().item(),
              '\n-Classify By Elbo:', vaes_compare.float().mean().item())

        self.uncertainty_analysis(vaes_scores, classifier_scores, targets, combined_scores)

        ''' 
                F1 score
        '''

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

    def uncertainty_analysis(self, vaes_scores, classifier_scores, targets, combined_scores):

        _, combined_predictions = combined_scores.max(dim=-1)
        _, classifier_predictions = classifier_scores.max(dim=-1)
        _, vaes_predictions = vaes_scores.max(dim=-1)

        classifier_compare = classifier_predictions.eq(targets)
        combined_compare = combined_predictions.eq(targets)
        vaes_compare = vaes_predictions.eq(targets)

        '''
                        uncertainty analyses
        '''

        vaes_scores_softmax = nn.Softmax(dim=-1)(vaes_scores)
        classifier_predictions_indices, _ = classifier_scores.max(dim=-1)
        classifier_prediction_values = classifier_scores[np.arange(0, len(classifier_scores)),
                                                         classifier_predictions_indices.long()]

        classifier_uncertain_indices = ((classifier_prediction_values < 0.50).eq(
            classifier_prediction_values > 0.00)).nonzero()

        # vae_scores_for_uncertain = vaes_scores[classifier_uncertain_indices]
        vae_scores_for_uncertain, pred_vae = vaes_scores_softmax[classifier_uncertain_indices.long()].max(dim=-1)
        classifier_uncertain_scores, pred_class = classifier_scores[classifier_uncertain_indices.long()].max(dim=-1)
        true = targets[classifier_uncertain_indices.long()]

        print('LSTM is uncertain in', len(classifier_uncertain_indices)/len(classifier_scores), 'samples.')
        classifier_uncertain_indices_correct = classifier_compare[classifier_uncertain_indices].nonzero()
        classifier_uncertain_indices_false = (classifier_compare[classifier_uncertain_indices]==0).nonzero()
        print('-', len(classifier_uncertain_indices_false)/len(classifier_uncertain_indices), 'of these are misclassifications.')

        classifier_uncertain_correct_VAE = vaes_compare[classifier_uncertain_indices_correct]
        classifier_uncertain_false_VAE = vaes_compare[classifier_uncertain_indices_false]

        print('- -', classifier_uncertain_correct_VAE.float().mean().item(), 'of the CORRECT uncertain classifications are correctly classified by the VAE.')
        print('- -', classifier_uncertain_false_VAE.float().mean().item(), 'of the uncertain MISclassifications are correctly classified by the VAE.')

        classifier_uncertain_correct_Combined = combined_compare[classifier_uncertain_indices_correct]
        classifier_uncertain_false_Combined = combined_compare[classifier_uncertain_indices_false]

        print('- - -', classifier_uncertain_correct_Combined.float().mean().item(),
              'of the CORRECT uncertain classifications are correctly classified by the Combined Model.')
        print('- - -', classifier_uncertain_false_Combined.float().mean().item(),
              'of the uncertain MISclassifications are correctly classified by the Combined Model.')

        print('cla', classifier_uncertain_scores.tolist())
        print('vae', vae_scores_for_uncertain.tolist())
        print('cla', pred_class.tolist())
        print('vae', pred_vae.tolist())
        print('tru', true.tolist())

