import os, torch, sys
from utils.system_utils import setup_directories, save_codebase_of_run
from utils.model_utils import calculate_accuracy
from torch.utils.data import DataLoader
from utils.constants import *
from typing import List, Tuple
from models.enums.Genre import Genre

class Tester:
    # input: both network models
    # return average loss, acc; etc.

    def __init__(self,
                 model,
                 data_loader_test: DataLoader,
                 data_loader_sentence,
                 model_state_path='',
                 device='cpu'):

        # the saved network as an object
        self.model = model
        self.model_state_path = model_state_path
        self.data_loader_test = data_loader_test
        self.data_loader_sentence = data_loader_sentence
        self.device = device
        self.model.eval()

    def test(self):
        """
         main training function
        """
        # setup data output directories:
        setup_directories()
        # save_codebase_of_run(self.arguments)

        try:
            # loading saved trained weights
            if self.model_state_path:  # because if we are testing a CombinedClassifier the states are already loaded
                self.model.load_state_dict(torch.load(self.model_state_path))

            log = {'final_scores': [], 'combination': {
                'classifier_scores': [], 'vaes_scores': []},
                   'accuracies_per_batch': [],
                   'true_targets': []}

            # for i, items in enumerate(self.data_loader_test):
            for i, items in enumerate(zip(self.data_loader_test, self.data_loader_sentence)):
                (batch, targets, lengths), (batch2, targets2, lengths2) = items
                # if 'Wrapper' in type(self.data_loader_test.dataset).__name__:
                #     (batch, targets, lengths), (batch2, targets2, lengths2) = items
                # else:
                #     (batch, targets, lengths) = items

                accuracy_batch = self._batch_iteration(batch, targets, lengths, log, (batch2, targets2, lengths), i)

                log['accuracies_per_batch'].append(accuracy_batch)
                log['true_targets'].append(targets)
                # break
                # if i>2:
                #     break
            # average over all accuracy batches
            # batches_tested = len(results)
            # average_accuracy = torch.mean(results)
            # average_scores = torch.mean(results["acc"])/batches_tested

            # return average_accuracy, average_scores
            return log
        except KeyboardInterrupt as e:
            print(f"Killed by user: {e}")

            return False
        except Exception as e:
            print(e)

            raise e

    def _batch_iteration(self,
                         batch: torch.Tensor,
                         targets: torch.Tensor,
                         lengths: torch.Tensor,
                         log,
                         sentencebatch, step):
        """
        runs forward pass on batch and backward pass if in train_mode
        """
        batch2, targets2, lengths2 = sentencebatch
        batch = batch.to(self.device).detach()
        targets = targets.to(self.device).detach()
        lengths = lengths.to(self.device).detach()

        if batch2 is not None:
            batch2 = batch2.to(self.device).detach()
            targets2 = targets2.to(self.device).detach()
            lengths2 = lengths2.to(self.device).detach()

        output = self.model.forward(batch, targets, lengths, (batch2, targets2, lengths2), step)

        final_scores_per_class = output
        if 'Combined' in type(self.model).__name__:
            final_scores_per_class, (score_classifier, score_elbo) = output
            log['combination']['classifier_scores'].append(score_classifier.detach())
            log['combination']['vaes_scores'].append(score_elbo.detach())

        _, classifications = score_classifier.detach().max(dim=-1)
        accuracy = (targets.eq(classifications)).float().mean().item()

        log['final_scores'].append(final_scores_per_class.detach())
        return accuracy
