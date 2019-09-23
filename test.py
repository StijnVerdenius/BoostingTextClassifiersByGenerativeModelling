import os
import torch
import sys
from utils.system_utils import setup_directories, save_codebase_of_run

from utils.model_utils import calculate_accuracy
class Tester:
    # input: both network models
    # return average loss, acc; etc.
    #

    def __init__(self, model, model_state_path, data_loader_test):
        # the saved network as an object
        self.model = model
        self.model_state_path = model_state_path
        self.data_loader_test = data_loader_test


    def test(self):
        """
         main training function
        """

        # setup data output directories:
        setup_directories()
        # save_codebase_of_run(self.arguments)

        # data gathering
        try:

            # loading saved trained weights
            self.model.load_state_dict(torch.load(self.model_state_path))

            results = []
            for i, (batch, targets, lengths) in enumerate(self.data_loader_test):

                # do forward pass and whatnot on batch
                loss_batch, accuracy_batch = self._batch_iteration(batch, targets, lengths)

                # add to list somehow: todo: make statistic class?
                results.append({"loss": loss_batch, "acc": accuracy_batch})


            # average over all accuracy batches
            batches_tested = len(results["loss"])
            average_accuracy = torch.mean(results["loss"])/batches_tested
            average_scores = torch.mean(results["acc"])/batches_tested

            return average_accuracy, average_scores

        except KeyboardInterrupt as e:
            print(f"Killed by user: {e}")

            return False
        except Exception as e:
            print(e)

            raise e


