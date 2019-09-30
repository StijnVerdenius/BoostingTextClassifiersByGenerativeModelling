from train import *


class JointTraining(Trainer):

    def __init__(self, data_loader_train_1: DataLoader, model: GeneralModel,
                 optimizer: Optimizer, loss_function: GeneralModel, args: argparse.Namespace, patience: int,
                 data_loader_train_2: DataLoader, device="cpu"):
        super().__init__(data_loader_train_1, None, model, optimizer, loss_function, args, patience, device=device)
        self.model.combination_method = "learn"
        self.data_loader_train_2 = data_loader_train_2

    def _epoch_iteration(
            self,
            epoch_num: int,
            best_metrics: Tuple[float, float],
            patience: int) -> Tuple[List, Tuple, int]:


        progress = []

        train_accuracy = 0
        train_loss = 0
        data_loader_length = len(self.data_loader_train)

        for i, items in enumerate(zip(self.data_loader_train, self.data_loader_train_2)):
            (batch, targets, lengths), (batch2, targets2, lengths2) = items
            print(f'Train: {i}/{data_loader_length}       \r', end='')

            # do forward pass and whatnot on batch
            loss_batch, accuracy_batch = self._batch_iteration_joint(batch, targets, lengths, (batch2, targets2, lengths2), i)
            train_loss += loss_batch
            train_accuracy += accuracy_batch

            # add to list somehow:
            progress.append({"loss": loss_batch, "acc": accuracy_batch})

            # calculate amount of batches and walltime passed
            time_passed = datetime.now() - DATA_MANAGER.actual_date

            # check if runtime is expired
            if (time_passed.total_seconds() > (self.arguments.max_training_minutes * 60)) \
                    and self.arguments.max_training_minutes > 0:
                raise KeyboardInterrupt(f"Process killed because {self.arguments.max_training_minutes} minutes passed "
                                        f"since {DATA_MANAGER.actual_date}. Time now is {datetime.now()}")

            if patience == 0:
                break

        return progress, best_metrics, patience

    def _batch_iteration_joint(self,
                               batch: torch.Tensor,
                               targets: torch.Tensor,
                               lengths: torch.Tensor,
                               sentencebatch,
                               step):
        """
        runs forward pass on batch and backward pass if in train_mode
        """

        batch2, targets2, lengths2 = sentencebatch
        batch = batch.to(self._device)
        targets = targets.to(self._device)
        lengths = lengths.to(self._device)

        self.model.train()
        self.optimizer.zero_grad()

        if batch2 is not None:
            batch2 = batch2.to(self._device)
            targets2 = targets2.to(self._device)
            lengths2 = lengths2.to(self._device)

        output, (_, _) = self.model.forward(batch, targets, lengths, (batch2, targets2, lengths2), step)

        loss = self.loss_function(targets, output)
        accuracy = calculate_accuracy(targets, output)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        loss = loss.item()
        accuracy = accuracy.item()

        if (step % 100) == 0:
            print(step, loss, accuracy)

        return loss, accuracy
