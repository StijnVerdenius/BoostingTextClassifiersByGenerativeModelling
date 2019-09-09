from typing import List

from utils.constants import *
from tensorboardX import SummaryWriter

import sys, os

from utils.model_utils import save_models
from utils.system_utils import setup_directories, save_codebase_of_run

from datetime import datetime


class Trainer:

    def __init__(self,
                 data_loader,
                 model,
                 optimizer,
                 loss_function,
                 args
                 ):

        self.arguments = args
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model = model
        self.data_loader = data_loader

        # init current runs timestamp
        DATA_MANAGER.set_date_stamp()

        # initialize tensorboardx
        self.writer = SummaryWriter(
            f"{GITIGNORED_DIR}/{RESULTS_DIR}/{DATA_MANAGER.stamp}/tensorboardx/")

        # todo: check input on nulls

    def train(self):
        """
         main training function

        :return:
        """

        # setup data output directories:
        setup_directories()
        save_codebase_of_run(self.arguments)

        # data gathering
        progress = []

        epoch = 0

        try:

            print(f"{PRINTCOLOR_BOLD}Started training with the following config:{PRINTCOLOR_END}\n{self.arguments}")

            # run
            for epoch in range(self.arguments.epochs):

                print(
                    f"\n\n{PRINTCOLOR_BOLD}Starting epoch{PRINTCOLOR_END} {epoch}/{self.arguments.epochs} at {str(datetime.now())}")

                # do epoch
                epoch_progress = self._epoch_iteration(epoch)

                # add progress
                progress += epoch_progress

                # write progress to pickle file (overwrite because there is no point keeping seperate versions)
                DATA_MANAGER.save_python_obj(progress, f"{DATA_MANAGER.stamp}/{PROGRESS_DIR}/progress_list",
                                             print_success=False)

                # write models if needed (don't save the first one
                if (((epoch + 1) % self.arguments.saving_freq) == 0):
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

        for i, batch in enumerate(self.data_loader):

            loss_batch, accuracy_batch = self._batch_iteration(batch)

            # calculate amount of passed batches
            batches_passed = i + (epoch_num * len(self.data_loader))

            # print progress to terminal
            if (batches_passed % self.arguments.eval_freq == 0):
                pass
                # todo

            # check if runtime is expired
            time_passed = datetime.now() - DATA_MANAGER.actual_date
            if (time_passed.total_seconds() > (self.arguments.max_training_minutes * 60)) \
                    and self.arguments.max_training_minutes > 0:
                raise KeyboardInterrupt(f"Process killed because {self.arguments.max_training_minutes} minutes passed "
                                        f"since {DATA_MANAGER.actual_date}. Time now is {datetime.now()}")

        return progress

    def _batch_iteration(self, batch):

        output = self.model.forward(batch)

        loss = self.loss_function.forward(output)

        # etc

        return None, None  # todo
