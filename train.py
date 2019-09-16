import argparse
import sys
from datetime import datetime
from typing import List, Tuple

import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from models import GeneralModel
from utils.constants import *
from utils.model_utils import save_models, calculate_accuracy, delete_list, assert_type
from utils.system_utils import setup_directories, save_codebase_of_run


class Trainer:

    def __init__(self,
                 data_loader_train: DataLoader,
                 data_loader_validation: DataLoader,
                 model: GeneralModel,
                 optimizer: Optimizer,
                 loss_function: GeneralModel,
                 args: argparse.Namespace
                 ):

        self.arguments = args
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model = model
        self.data_loader_validation = data_loader_validation
        self.data_loader_train = data_loader_train

        # validate input to class
        self._validate_self()

        # init current runs timestamp
        DATA_MANAGER.set_date_stamp(addition=args.run_name)

        # initialize tensorboardx
        self.writer = SummaryWriter(os.path.join(GITIGNORED_DIR, RESULTS_DIR, DATA_MANAGER.stamp, SUMMARY_DIR))

    def _validate_self(self):
        # assert_type(self.arguments, argparse.Namespace)
        # assert_type(self.loss_function, GeneralModel)
        # assert_type(self.model, GeneralModel)
        # assert_type(self.data_loader_validation, DataLoader)
        # assert_type(self.data_loader_train, DataLoader)
        # assert_type(self.optimizer, Optimizer)
        pass


    def train(self) -> bool:
        """
         main training function
        """

        # setup data output directories:
        setup_directories()
        save_codebase_of_run(self.arguments)

        # data gathering
        progress = []

        epoch = 0

        try:

            print(f"{PRINTCOLOR_BOLD}Started training with the following config:{PRINTCOLOR_END}\n{self.arguments}\n\n")

            # run
            for epoch in range(self.arguments.epochs):

                print(
                    f"\n\n{PRINTCOLOR_BOLD}Starting epoch{PRINTCOLOR_END} {epoch}/{self.arguments.epochs} at {str(datetime.now())}")

                # do epoch
                epoch_progress = self._epoch_iteration(epoch)

                # add progress-list to global progress-list
                progress += epoch_progress

                # write progress to pickle file (overwrite because there is no point keeping seperate versions)
                DATA_MANAGER.save_python_obj(progress,
                                             os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, PROGRESS_DIR,
                                                          "progress_list"),
                                             print_success=False)

                # write models if needed (don't save the first one
                if ((epoch + 1) % self.arguments.saving_freq) == 0:
                    save_models([self.model], f"Models_at_epoch_{epoch}")

                # flush prints
                sys.stdout.flush()

        except KeyboardInterrupt as e:
            print(f"Killed by user: {e}")
            save_models([self.model], f"KILLED_at_epoch_{epoch}")
            return False
        except Exception as e:
            print(e)
            save_models([self.model], f"CRASH_at_epoch_{epoch}")
            raise e

        # flush prints
        sys.stdout.flush()

        # example last save
        save_models([self.model], "finished")
        return True

    def _epoch_iteration(self, epoch_num: int) -> List:
        """
        one epoch implementation
        """

        progress = []

        for i, (batch, targets) in enumerate(self.data_loader_train):

            # do forward pass and whatnot on batch
            loss_batch, accuracy_batch = self._batch_iteration(batch, targets)

            # add to list somehow: todo: make statistic class?
            progress.append({"loss": loss_batch, "acc": accuracy_batch})

            # calculate amount of batches and walltime passed
            batches_passed = i + (epoch_num * len(self.data_loader_train))
            time_passed = datetime.now() - DATA_MANAGER.actual_date

            # run on validation set and print progress to terminal
            if (batches_passed % self.arguments.eval_freq) == 0:  # todo
                loss_validation, acc_validation = self._evaluate()
                self._log(loss_validation, acc_validation, loss_batch, accuracy_batch, batches_passed, float(time_passed.microseconds))


            # check if runtime is expired
            if (time_passed.total_seconds() > (self.arguments.max_training_minutes * 60)) \
                    and self.arguments.max_training_minutes > 0:
                raise KeyboardInterrupt(f"Process killed because {self.arguments.max_training_minutes} minutes passed "
                                        f"since {DATA_MANAGER.actual_date}. Time now is {datetime.now()}")

        return progress

    def _batch_iteration(self,
                         batch: torch.Tensor,
                         targets: torch.Tensor,
                         train_mode: bool = True) -> Tuple[float, float]:
        """
        runs forward pass on batch and backward pass if in train_mode
        """

        batch = batch.to(DEVICE)
        targets = targets.to(DEVICE)

        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        output = self.model.forward(batch)
        loss = self.loss_function.forward(targets, *output)

        if train_mode:
            loss.backward()
            self.optimizer.step()

        accuracy = 0
        if self.arguments.train_classifier:
            accuracy = calculate_accuracy(targets, *output).item()

        delete_list([output, batch, targets])

        return loss.item(), accuracy

    def _evaluate(self) -> Tuple[float, float]:
        """
        runs iteration on validation set
        """

        accs = []
        losses = []

        for i, (batch, targets) in enumerate(self.data_loader_validation):
            # do forward pass and whatnot on batch
            loss_batch, accuracy_batch = self._batch_iteration(batch, targets, train_mode=False)
            accs.append(accuracy_batch)
            losses.append(loss_batch)

        return float(np.mean(losses)), float(np.mean(accs))

    def _log(self,
             loss_validation: float,
             acc_validation: float,
             loss_train: float,
             acc_train: float,
             batches_done: int,
             time_passed:  float,
             ):
        """
        logs progress to user through tensorboard and terminal
        """
        self.writer.add_scalar("Accuracy_validation", acc_validation, batches_done, time_passed)
        self.writer.add_scalar("Loss_validation", loss_validation, batches_done, time_passed)
        self.writer.add_scalar("Loss_train", loss_train, batches_done, time_passed)
        self.writer.add_scalar("Accuracy_train", acc_train, batches_done, time_passed)
        print(f"Accuracy_validation: {acc_validation}, Loss_validation: {loss_validation}, Accuracy_train: {acc_train}, Loss_train: {loss_train}")


