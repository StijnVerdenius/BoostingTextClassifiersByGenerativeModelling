import torch

try:
    torch.cuda.current_device()
except:
    pass

import argparse
import sys
from torch.utils.data import DataLoader

from models.enums.Genre import Genre

from test import Tester
from train import Trainer
from utils.constants import *
from utils.model_utils import find_right_model
from utils.system_utils import ensure_current_directory
from utils.dataloader_utils import pad_and_sort_batch
import numpy as np
import random


def main(arguments: argparse.Namespace):
    """ where the magic happens """
    if DEVICE == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(SEED)

    # for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # get model from models-folder (name of class has to be identical to filename)
    model = find_right_model((CLASS_DIR if arguments.train_classifier else
                              (GEN_DIR if not arguments.combined_classification else CLASS_DIR)),
                             (arguments.classifier if arguments.train_classifier else
                              (arguments.generator if not arguments.combined_classification
                               else arguments.classifier)),
                             n_channels_in=arguments.embedding_size,
                             num_classes=arguments.num_classes,
                             hidden_dim=arguments.hidden_dim,
                             z_dim=arguments.z_dim,
                             device=DEVICE,
                             lstm_file=arguments.classifier_dir,
                             vae_files=arguments.vaes_dir,
                             input_dim=arguments.embedding_size)
    model.to(DEVICE)


    # if we are in train mode..
    if arguments.test_mode:
        # we are in test mode
        data_loader_test = load_data_set(arguments, TEST_SET)

        tester = Tester(model, data_loader_test)
        tester.test()
        pass  # todo: testing functionality, loading pretrained model
    else:

        # load needed data
        data_loader_train = load_data_set(arguments, TRAIN_SET)
        data_loader_validation = load_data_set(arguments, VALIDATION_SET)

        # get optimizer and loss function
        optimizer = find_right_model(OPTIMS, arguments.optimizer, params=model.parameters(), lr=arguments.learning_rate)
        loss_function = find_right_model(LOSS_DIR, arguments.loss).to(DEVICE)

        # train
        trainer = Trainer(data_loader_train, data_loader_validation, model, optimizer, loss_function, arguments)
        trainer.train()


def load_data_set(arguments: argparse.Namespace,
                  set_name: str) -> DataLoader:
    """ loads specific dataset as a DataLoader """

    dataset = find_right_model(DATASETS, arguments.dataset_class, folder=arguments.data_folder, set_name=set_name, genre=Genre.from_str(arguments.genre))
    loader = DataLoader(dataset, shuffle=(set_name is TRAIN_SET), batch_size=arguments.batch_size, collate_fn=pad_and_sort_batch)

    # todo: revisit and validation checks
    return loader


def parse() -> argparse.Namespace:
    """ does argument parsing """

    parser = argparse.ArgumentParser()

    # int
    parser.add_argument('--epochs', default=500, type=int, help='max number of epochs')
    parser.add_argument('--eval_freq', default=20, type=int, help='evaluate every x batches')
    parser.add_argument('--saving_freq', default=1, type=int, help='save every x epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='size of batches')
    parser.add_argument('--embedding_size', default=256, type=int, help='size of embeddings')  # todo
    parser.add_argument('--num_classes', default=5, type=int, help='size of embeddings')  # todo
    parser.add_argument('--hidden_dim', default=128, type=int, help='size of batches')
    parser.add_argument('--z_dim', default=128, type=int, help='size of batches')
    parser.add_argument('--max_training_minutes', default=24 * 60, type=int,
                        help='max mins of training be4 save-and-kill')

    # float
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')

    # string
    parser.add_argument('--classifier', default="LSTMClassifier", type=str, help='classifier model name')
    parser.add_argument('--generator', default="BaseVAE", type=str, help='generator model name')
    parser.add_argument('--loss', default="CrossEntropyLoss", type=str, help='loss-function model name')
    parser.add_argument('--optimizer', default="Adam", type=str, help='optimizer model name')
    parser.add_argument('--data_folder', default=os.path.join('local_data', 'data'), type=str, help='data folder path')
    parser.add_argument('--dataset_class', default="LyricsDataset", type=str, help='dataset name')
    parser.add_argument('--run_name', default="", type=str, help='extra identification for run')
    parser.add_argument('--genre', type=str, default=None,
                        help='vae-genre')
    # parser.add_argument('--genre', default=None, type=Genre, help='vae-genre')

    # bool
    parser.add_argument('--test-mode', action='store_true', help='start in train_mode')
    parser.add_argument('--train-classifier', action='store_true', help='train a classifier')
    parser.add_argument('--combined_classification', action='store_true', help='combined classification')

    # combined test stuff
    parser.add_argument('--classifier_dir', default="", type=str, help='classifier state-dict dir')
    parser.add_argument('--vaes_dir', default="", type=str, help='vaes state-dict dir. Give names separated by commas')

    # todo: add whatever you like

    return parser.parse_args()


if __name__ == '__main__':
    print("PyTorch version:", torch.__version__, "Python version:", sys.version)
    print("Working directory: ", os.getcwd())
    print("CUDA avalability:", torch.cuda.is_available(), "CUDA version:", torch.version.cuda)
    ensure_current_directory()
    args = parse()
    main(args)
