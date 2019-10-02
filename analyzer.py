import os, torch, sys
from utils.system_utils import setup_directories, save_codebase_of_run
from utils.model_utils import calculate_accuracy
from torch.utils.data import DataLoader
from utils.constants import *
import pickle
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from typing import List
from plots import save_percentage_plot
import numpy as np
from torch import nn

from statsmodels.stats.contingency_tables import mcnemar
# from mlxtend.evaluate import permutation_test


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

    def calculate_mcnemars_test(self, targets, predictions1, predictions2):
        contingency_table = self.create_contingency_table(
            targets,
            predictions1,
            predictions2)
        
        result = mcnemar(contingency_table, exact=True)
        return result.pvalue

    def calculate_confusion_matrix(
        self, 
        targets,
        predictions,
        classes,
        analysis_folder,
        normalize=False,
        plot_matrix=True,
        title=None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        # Compute confusion matrix
        cm = metrics.confusion_matrix(targets, predictions)
        # Only use the labels that appear in the data
        labels = unique_labels(targets, predictions)
        classes = classes[labels]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        ax = None
        if plot_matrix:
            ax = self.plot_confusion_matrix(cm, classes, analysis_folder, normalize, title)

        return cm, ax

    def plot_confusion_matrix(
        self,
        cm,
        classes,
        analysis_folder,
        normalize=False,
        title=None,
        print_scores=True,
        cmap=plt.cm.Blues):

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        ax.set_ylim(4.5, -0.5) # fix the classes

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        
        if print_scores:
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
        fig.tight_layout()
        fig.savefig(os.path.join(analysis_folder, f'confusion_matrix_{title}'))

        return ax

    def compute_confusion_matrix(
        self,
        targets,
        combined_predictions,
        classifier_predictions,
        analysis_folder):

        classes=np.array(['Pop', 'Hip-Hop', 'Rock', 'Metal', 'Country'])
        combined_cm, _ = self.calculate_confusion_matrix(targets, combined_predictions, classes, analysis_folder, normalize=False, title='Combined')
        lstm_cm, _ = self.calculate_confusion_matrix(targets, classifier_predictions, classes, analysis_folder, normalize=False, title='LSTM')

        diff_cm = combined_cm - lstm_cm
        ones = np.ones(diff_cm.shape, dtype=np.int32) * (-1)
        ones += np.eye(diff_cm.shape[0], dtype=np.int32) * 2
        diff_cm = ones * diff_cm

        self.plot_confusion_matrix(
            diff_cm,
            classes,
            analysis_folder,
            normalize=False,
            title='Difference',
            cmap=plt.cm.RdYlGn,
            print_scores=False)

        plt.show()

    def compute_significance(self, targets, combined_predictions, classifier_predictions):
        mcnemars_p_value = self.calculate_mcnemars_test(targets, classifier_predictions, combined_predictions)
        alpha_value = 0.05
        mcnemars_significant = mcnemars_p_value < alpha_value
        print(f'Mcnemars: {mcnemars_significant} | p-value: {mcnemars_p_value}')

    def compute_f1(self, targets, combined_predictions, classifier_predictions, vaes_predictions):
        combined_f1, combined_precision, combined_recall = self.calculate_metrics(targets, combined_predictions)
        classifier_f1, classifier_precision, classifier_recall = self.calculate_metrics(targets, classifier_predictions)
        vae_f1, vae_precision, vae_recall = self.calculate_metrics(targets, vaes_predictions)

        print(f'Combined F1: {combined_f1}\nLSTM F1: {classifier_f1}\nVAE F1: {vae_f1}')

    def ensure_analyzer_filesystem(self):
        analysis_folder = os.path.join('local_data', 'analysis')
        if not os.path.exists(analysis_folder):
            os.mkdir(analysis_folder)

        return analysis_folder

    def analyze_misclassifications(self, test_logs):

        if test_logs is not None:
            with open('logs_full_on_full_wlens.pickle', 'wb') as handle:
                pickle.dump(test_logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('logs_full_on_full_wlens.pickle', 'rb') as handle:
                test_logs = pickle.load(handle)

        analysis_folder = self.ensure_analyzer_filesystem()

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

        print("----------------------------------------------")
        self.compute_f1(targets, combined_predictions, classifier_predictions, vaes_predictions)

        print("----------------------------------------------")
        self.compute_significance(targets, combined_predictions, classifier_predictions)

        print("----------------------------------------------")
        self.compute_confusion_matrix(targets, combined_predictions, classifier_predictions, analysis_folder)
        
        # check if combination correctly classified these? check how many
        # print(combined_compare[classifier_misfire_indices])

        # print(classifier_misfire_indices)

        # PLOT

        classifier_misfire_indices = (classifier_compare == 0).nonzero()  # get misclassifications
        combined_misfire_indices = (combined_compare == 0).nonzero()  # get misclassifications
        vaes_misfire_indices = (vaes_compare == 0).nonzero()  # get misclassifications

        len_of_dataset = len(classifier_compare.tolist())

        #  Compare LSTM with VAE
        vae_right_class_wrong = vaes_compare[classifier_misfire_indices].tolist().count([1]) / len_of_dataset
        vae_wrong_class_wrong = classifier_compare[vaes_misfire_indices].tolist().count([0]) / len_of_dataset
        vae_wrong_class_right = classifier_compare[vaes_misfire_indices].tolist().count([1]) / len_of_dataset

        #  Compare LSTM with Combined
        comb_right_class_wrong = combined_compare[classifier_misfire_indices].tolist().count([1]) / len_of_dataset
        comb_wrong_class_wrong = classifier_compare[combined_misfire_indices].tolist().count([0]) / len_of_dataset
        comb_wrong_class_right = classifier_compare[combined_misfire_indices].tolist().count([1]) / len_of_dataset

        lstm_classifier = classifier_compare.tolist().count(1) / len_of_dataset

        save_percentage_plot([lstm_classifier, 1 - lstm_classifier, 0, 0],
                       [1 - vae_wrong_class_wrong - vae_wrong_class_right -
                        vae_right_class_wrong, vae_wrong_class_right,
                        vae_right_class_wrong, vae_wrong_class_wrong],
                       [1 - comb_wrong_class_wrong - comb_wrong_class_right -
                        comb_right_class_wrong, comb_wrong_class_right,
                        comb_right_class_wrong, comb_wrong_class_wrong],
                       'Ipek_plot')


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

        classifier_uncertain_indices = ((classifier_prediction_values <= 0.35).eq(
            classifier_prediction_values >= 0.00)).nonzero()

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

        # print('cla', classifier_uncertain_scores.tolist())
        # print('vae', vae_scores_for_uncertain.tolist())
        # print('cla', pred_class.tolist())
        # print('vae', pred_vae.tolist())
        # print('tru', true.tolist())

