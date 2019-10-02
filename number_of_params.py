import operator
from functools import reduce
import torch

print(torch.cuda.is_available())
from utils.constants import *

basepath = os.path.join(GITIGNORED_DIR, RESULTS_DIR)
datamanager = DataManager(basepath)

paths_per_model = {
    "VAE": os.path.join("full_vae", "rock", "models", "finished"),
    "LSTM": os.path.join("full_lstm", "models", "model_best")
}

loaded_models = {key: datamanager.load_python_obj(value) for key, value in paths_per_model.items()}

combinations = {
    "VAE": [["VAE", 5]],
    "LSTM": [["LSTM"]],
    "COMBINED": [["VAE", 5], ["LSTM"]],
    "ENSEMBLE": [["VAE", 5], ["LSTM"], [10]]
}


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def get_summable(to_multiply):
    multiplied = 1
    for element in to_multiply:
        if (isinstance(element, str)):
            tensors = [(key, [val for val in value.values()]) for key, value in loaded_models[element].items()]
            element = sum([prod(x[1][0].shape) for x in tensors])
        multiplied *= element
    return multiplied


def get_number_of_params(to_sum):
    return sum([get_summable(sublist) for sublist in to_sum])


lengths = {
    key: get_number_of_params(value) for key, value in combinations.items()
}

print(lengths)
