import torch

from utils.data_manager import DataManager
import os

CODEBASE_DIR = "codebase"
SUMMARY_DIR = "summary"
OUTPUT_DIR = "output"
RESULTS_DIR = "results"
MODELS_DIR = "models"
PROGRESS_DIR = "progress"
DATA_DIR = "data"

PROJ_NAME = "DL4NLP"
GITIGNORED_DIR = "local_data"

DATA_MANAGER = DataManager(os.path.join(".", GITIGNORED_DIR))

OUTPUT_DIRS = [OUTPUT_DIR, SUMMARY_DIR, CODEBASE_DIR, MODELS_DIR, PROGRESS_DIR]

# printing
PRINTCOLOR_PURPLE = '\033[95m'
PRINTCOLOR_CYAN = '\033[96m'
PRINTCOLOR_DARKCYAN = '\033[36m'
PRINTCOLOR_BLUE = '\033[94m'
PRINTCOLOR_GREEN = '\033[92m'
PRINTCOLOR_YELLOW = '\033[93m'
PRINTCOLOR_RED = '\033[91m'
PRINTCOLOR_BOLD = '\033[1m'
PRINTCOLOR_UNDERLINE = '\033[4m'
PRINTCOLOR_END = '\033[0m'

LOSS_DIR = "losses"
CLASS_DIR = "classifiers"
GEN_DIR = "generators"
OPTIMS = "optim"
DATASETS = "datasets"

TEST_SET = "test"
VALIDATION_SET = "validation"
TRAIN_SET = "train"
