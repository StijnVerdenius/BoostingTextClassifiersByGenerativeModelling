import os, torch, sys
from utils.system_utils import setup_directories, save_codebase_of_run
from utils.model_utils import calculate_accuracy
from torch.utils.data import DataLoader
from utils.constants import *
from typing import List, Tuple


class Tester:
    # input: both network models
    # return average loss, acc; etc.

    def __init__(self,
                 model,
                 data_loader_test: DataLoader,
                 model_state_path='',):

        # the saved network as an object
        self.model = model
        self.model_state_path = model_state_path
        self.data_loader_test = data_loader_test

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

            results = []
            for i, (batch, targets, lengths) in enumerate(self.data_loader_test):

                accuracy_batch = self._batch_iteration(batch, targets, lengths)

                results.append(accuracy_batch)
                break
            # average over all accuracy batches
            # batches_tested = len(results)
            # average_accuracy = torch.mean(results)
            # average_scores = torch.mean(results["acc"])/batches_tested

            # return average_accuracy, average_scores

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
                         ):
        """
        runs forward pass on batch and backward pass if in train_mode
        """

        batch = batch.to(DEVICE).detach()
        targets = targets.to(DEVICE).detach()
        lengths = lengths.to(DEVICE).detach()
        print(batch.shape, lengths.shape)

        output = self.model.forward(batch, lengths)

        if 'Combined' in type(self.model).__name__:
            combined_score, scores = output
            score_classifier, _, _ = scores

        _, classifications = score_classifier.detach().max(dim=-1)
        accuracy = (targets.eq(classifications)).float().mean().item()

        return accuracy
