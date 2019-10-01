from train import *


class JointTraining(Trainer):

    def __init__(self, data_loader_train_1: DataLoader, data_loader_val_1: DataLoader, model: GeneralModel,
                 optimizer: Optimizer, loss_function: GeneralModel, args: argparse.Namespace, patience: int,
                 data_loader_train_2: DataLoader, data_loader_val_2: DataLoader, device="cpu"):
        super().__init__(data_loader_train_1, data_loader_val_1, model, optimizer, loss_function, args, patience, device=device)
        self.data_loader_validation_2 = data_loader_val_2
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
            batches_passed = i + (epoch_num * len(self.data_loader_train))

            # run on validation set and print progress to terminal
            # if we have eval_frequency or if we have finished the epoch
            if (batches_passed % self.arguments.eval_freq) == 0 or (i + 1 == data_loader_length):
                loss_validation, acc_validation = self._evaluate()

                new_best = False
                if self.model.compare_metric(best_metrics, loss_validation, acc_validation):
                    save_models([self.model], 'model_best')
                    best_metrics = (loss_validation, acc_validation)
                    new_best = True
                    patience = self._patience
                else:
                    patience -= 1

                self._log(
                    loss_validation,
                    acc_validation,
                    (train_loss / (i + 1)),
                    (train_accuracy / (i + 1)),
                    batches_passed,
                    float(time_passed.microseconds),
                    epoch_num,
                    i,
                    data_loader_length,
                    new_best)

                if (self.model.combination_method == "learn_sum"):
                    print("weights lstm:", self.model.W_classifier, "weights vaes:", self.model.W_vaes, "\n\n")

            # check if runtime is expired
            if (time_passed.total_seconds() > (self.arguments.max_training_minutes * 60)) \
                    and self.arguments.max_training_minutes > 0:
                raise KeyboardInterrupt(f"Process killed because {self.arguments.max_training_minutes} minutes passed "
                                        f"since {DATA_MANAGER.actual_date}. Time now is {datetime.now()}")

            if patience == 0:
                break

        return progress, best_metrics, patience

    def _evaluate(self) -> Tuple[float, float]:
        """
        runs iteration on validation set
        """

        accuracies = []
        losses = []
        data_loader_length = len(self.data_loader_validation)

        for i, items in enumerate(zip(self.data_loader_validation, self.data_loader_validation_2)):
            (batch, targets, lengths), (batch2, targets2, lengths2) = items
            print(f'Validation: {i}/{data_loader_length}       \r', end='')

            # do forward pass and whatnot on batch
            loss_batch, accuracy_batch = self._batch_iteration_joint(batch, targets, lengths, (batch2, targets2, lengths2), i, train_mode=False)
            accuracies.append(accuracy_batch)
            losses.append(loss_batch)

        return float(np.mean(losses)), float(np.mean(accuracies))


    def _batch_iteration_joint(self,
                               batch: torch.Tensor,
                               targets: torch.Tensor,
                               lengths: torch.Tensor,
                               sentencebatch,
                               step,
                               train_mode=True):
        """
        runs forward pass on batch and backward pass if in train_mode
        """

        batch2, targets2, lengths2 = sentencebatch
        batch = batch.to(self._device)
        targets = targets.to(self._device)
        lengths = lengths.to(self._device)

        if train_mode:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        if batch2 is not None:
            batch2 = batch2.to(self._device)
            targets2 = targets2.to(self._device)
            lengths2 = lengths2.to(self._device)

        output, (_, _) = self.model.forward(batch, targets, lengths, (batch2, targets2, lengths2), step)

        loss = self.loss_function(targets, output)
        accuracy = calculate_accuracy(targets, output)

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

        loss = loss.item()
        accuracy = accuracy.item()

        return loss, accuracy
