import torch

import argparse
import os
import sys
from torch.utils.data import DataLoader

from test import Tester
from train import Trainer
from utils.constants import *
from utils.model_utils import find_right_model
from utils.system_utils import ensure_current_directory


def main(arguments: argparse.Namespace):
    """ where the magic happens """

    # get model from models-folder (name of class has to be identical to filename)
    model = find_right_model((CLASS_DIR if arguments.train_classifier else GEN_DIR),
                             (arguments.classifier if arguments.train_classifier else arguments.generator),
                             n_channels_in=arguments.embedding_size,
                             num_classes=arguments.num_classes,
                             )

    # if we are in train mode..
    if arguments.train_mode:

        # load needed data
        data_loader_train = load_data_set(arguments, TRAIN_SET)
        data_loader_validation = load_data_set(arguments, VALIDATION_SET)

        # get optimizer and loss function
        optimizer = find_right_model(OPTIMS, arguments.optimizer, params=model.parameters(), lr=arguments.learning_rate)
        loss_function = find_right_model(LOSS_DIR, arguments.loss, some_param="example")

        # train
        trainer = Trainer(data_loader_train, data_loader_validation, model, optimizer, loss_function, arguments)
        trainer.train()

    else:
        # we are in test mode
        data_loader_test = load_data_set(arguments, TEST_SET)

        tester = Tester()
        raise NotImplementedError
        pass  # todo: testing functionality, loading pretrained model


def load_data_set(arguments: argparse.Namespace,
                  set_name: str) -> DataLoader:
    """ loads specific dataset as a DataLoader """

    dataset = find_right_model(DATASETS, arguments.data_class, file=arguments.data_file, set_name=set_name)
    loader = DataLoader(dataset, shuffle=True, batch_size=arguments.batch_size, drop_last=True)
    # todo: revisit and validation checks
    return loader


def parse() -> argparse.Namespace:
    """ does argument parsing """

    parser = argparse.ArgumentParser()

    # int
    parser.add_argument('--epochs', default=500, type=int, help='max number of epochs')
    parser.add_argument('--eval_freq', default=20, type=int, help='evaluate every x batches')
    parser.add_argument('--saving_freq', default=50, type=int, help='save every x epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='size of batches')
    parser.add_argument('--embedding_size', default=100, type=int, help='size of embeddings') # todo
    parser.add_argument('--num_classes', default=10, type=int, help='size of embeddings') # todo
    # parser.add_argument('--hidden_size', default=100, type=int, help='size of batches')
    # parser.add_argument('--z_size', default=100, type=int, help='size of batches')
    parser.add_argument('--max_training_minutes', default=24 * 60, type=int,
                        help='max mins of training be4 save-and-kill')

    # float
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')

    # string
    parser.add_argument('--classifier', default="LSTMClassifier", type=str, help='classifier model name')
    parser.add_argument('--generator', default="DummyGenerator", type=str, help='generator model name')
    parser.add_argument('--loss', default="CrossEntropyLoss", type=str, help='loss-function model name')
    parser.add_argument('--optimizer', default="Adam", type=str, help='optimizer model name')
    parser.add_argument('--data_file', default="data.file", type=str, help='data file name')
    parser.add_argument('--data_class', default="DummyDataLoader", type=str, help='dataloader model name')
    parser.add_argument('--run_name', default="", type=str, help='extra identification for run')

    # bool
    parser.add_argument('--train_mode', default=True, type=bool, help='start in train_mode')
    parser.add_argument('--train_classifier', default=True, type=bool, help='train a classifier')

    # todo: add whatever you like

    return parser.parse_args()


if __name__ == '__main__':
    print("PyTorch version:", torch.__version__, "Python version:", sys.version)
    print("Working directory: ", os.getcwd())
    print("CUDA avalability:", torch.cuda.is_available(), "CUDA version:", torch.version.cuda)
    ensure_current_directory()
    args = parse()
    main(args)
