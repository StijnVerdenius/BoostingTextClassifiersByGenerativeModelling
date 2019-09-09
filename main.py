import argparse

import torch
import os
import sys

from train import Trainer
from utils.constants import *
from utils.model_utils import find_right_model
from utils.system_utils import ensure_current_directory


def main(args):

    data_loader = load_data(args)

    model = find_right_model((CLASS_DIR if args.train_classifier else GEN_DIR),
                             (args.classifier if args.train_classifier else args.generator), some_param="example")

    if (args.train_mode):

        optimizer = find_right_model(OPTIMS, args.optimizer, params=model.parameters(), lr=args.learning_rate)
        loss_function = find_right_model(LOSS_DIR, args.loss, some_param="example")
        trainer = Trainer(data_loader, model, optimizer, loss_function, args)

        trainer.train()

    else:
        pass  # todo


def load_data(args):
    # todo: implement

    DATA_MANAGER.load_python_obj(f"/data/{args.data_file}")
    pass


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=500, type=int, help='max number of epochs')
    parser.add_argument('--eval_freq', default=10, type=int, help='evaluate every x epochs')
    parser.add_argument('--saving_freq', default=50, type=int, help='save every x epochs')

    parser.add_argument('--max_training_minutes', default=1e-4, type=int, help='max mins of training be4 save-and-kill')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')

    parser.add_argument('--classifier', default="DummyClassifier", type=str, help='classifier model name')
    parser.add_argument('--generator', default="DummyGenerator", type=str, help='generator model name')
    parser.add_argument('--loss', default="DummyLoss", type=str, help='loss-function model name')
    parser.add_argument('--optimizer', default="Adam", type=str, help='loss-function model name')
    parser.add_argument('--data_file', default="data.file", type=str, help='data file name')

    parser.add_argument('--train_mode', default=True, type=bool, help='start in train_mode')
    parser.add_argument('--train_classifier', default=True, type=bool, help='train a classifier')


    # todo: add

    return parser.parse_args()


if __name__ == '__main__':
    print("cuda_version:", torch.version.cuda, "pytorch version:", torch.__version__, "python version:", sys.version)
    print("Working directory: ", os.getcwd())
    ensure_current_directory()
    args = parse()
    main(args)
