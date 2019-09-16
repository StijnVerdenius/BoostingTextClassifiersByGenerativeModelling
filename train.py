import argparse
import sys
from datetime import datetime
from typing import List, Tuple

import torch
from tensorboardX import SummaryWriter
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from models import GeneralModel
from utils.constants import *
from utils.model_utils import save_models
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
        DATA_MANAGER.set_date_stamp()

        # initialize tensorboardx
        self.writer = SummaryWriter(
            f"{GITIGNORED_DIR}/{RESULTS_DIR}/{DATA_MANAGER.stamp.replace(':', '.')}/{SUMMARY_DIR}/")

    def _validate_self(self):
        # raise NotImplementedError
        # todo: validate all self-fields on nulls and correct types and filling
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
                                             f"{RESULTS_DIR}/{DATA_MANAGER.stamp}/{PROGRESS_DIR}/progress_list",
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

        for i, batch in enumerate(self.data_loader_train):

            # do forward pass and whatnot on batch
            loss_batch, accuracy_batch = self._batch_iteration(batch)

            # add to list somehow: todo: make statistic class?
            progress.append({"loss": loss_batch, "acc": accuracy_batch})

            # calculate amount of batches passed
            batches_passed = i + (epoch_num * len(self.data_loader_train))

            # run on validation set and print progress to terminal
            if (batches_passed % self.arguments.eval_freq == 0):  # todo
                loss_validation, acc_validation = self._evaluate()
                self._log(loss_validation, acc_validation)

            # check if runtime is expired
            time_passed = datetime.now() - DATA_MANAGER.actual_date
            if (time_passed.total_seconds() > (self.arguments.max_training_minutes * 60)) \
                    and self.arguments.max_training_minutes > 0:
                raise KeyboardInterrupt(f"Process killed because {self.arguments.max_training_minutes} minutes passed "
                                        f"since {DATA_MANAGER.actual_date}. Time now is {datetime.now()}")

        return progress

    def _batch_iteration(self,
                         batch: torch.Tensor,
                         train_mode: bool = True) -> Tuple[float, float]:
        """
        runs forward pass on batch and backward pass if in train_mode
        """

        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        print('IMHERE', batch[0].shape)
        batch_data, target = batch[0], batch[1]
        output = self.model.forward(batch_data)
        if 'LSTM' in type(self.model).__name__:
            output, (_, _) = output
        print('Output', output)
        loss = self.loss_function.forward((output, target))
        print('Loss', loss)

        raise NotImplementedError

        return None, None  # todo

    def _evaluate(self) -> Tuple[float, float]:
        """
        runs iteration on validation set
        """

        raise NotImplementedError
        return None, None  # todo

    def _log(self,
             loss_validation: float,
             acc_validation: float):
        """
        logs progress to user through tensorboard and terminal
        """

        raise NotImplementedError
        pass  # todo
