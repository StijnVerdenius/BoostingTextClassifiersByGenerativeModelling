from torch.utils.data import DataLoader
from utils.constants import *
from typing import List, Tuple
from models.enums.Genre import Genre
import numpy as np
from torch.utils.data import DataLoader

from utils.constants import *


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
         main testing function
        """

        try:
            # loading saved trained weights
            if self.model_state_path:  # because if we are testing a CombinedClassifier the states are already loaded
                self.model.load_state_dict(torch.load(self.model_state_path))

            log = {'final_scores': [], 'combination': {
                'classifier_scores': [], 'vaes_scores': []},
                   'accuracies_per_batch': [],
                   'true_targets': [],
                   'length_lstm': [],
                   'length_vae': []
                   }

            for i, items in enumerate(zip(self.data_loader_test, self.data_loader_sentence)):
                (batch, targets, lengths), (batch2, targets2, lengths2) = items

                accuracy_batch = self._batch_iteration(batch, targets, lengths, log, (batch2, targets2, lengths2), i)

                log['accuracies_per_batch'].append(accuracy_batch)
                log['true_targets'].append(targets)

                # if i % 100 == 0:
                    # print('combined accuracy so far', np.mean(log['accuracies_per_batch']), i)

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

        if (step % 100) == 0:
            print(step)

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
        log['length_lstm'].append(lengths.detach())
        log['length_vae'].append(lengths2.detach())

        return accuracy
