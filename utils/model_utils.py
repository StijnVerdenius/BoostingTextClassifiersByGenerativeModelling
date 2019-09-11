import importlib
import os
from typing import List

import torch.nn as nn
import torch.optim as opt

from utils.constants import *

types = [CLASS_DIR, GEN_DIR, LOSS_DIR, LOADERS]
models = {x: {} for x in types}


def _read_all_class_names():
    """
    private function that imports all class references in a dictionary
    """

    for typ in types:
        for name in os.listdir(f"./models/{typ}"):
            if not "__" in name:
                short_name = name.split(".")[0]
                module = importlib.import_module(f"models.{typ}.{short_name}")
                class_reference = getattr(module, short_name)
                models[typ][short_name] = class_reference

    models[OPTIMS] = {}
    models[OPTIMS]["Adam"] = opt.Adam
    models[OPTIMS]["RMSprop"] = opt.RMSprop
    models[OPTIMS]["SGD"] = opt.SGD


def find_right_model(type: str, name: str, **kwargs):
    """
    returns model with arguments given a string name-tag
    """

    return models[type][name](**kwargs)


def save_models(models: List[nn.Module],
                suffix: str):
    """
    Saves current state of models
    """

    save_dict = {str(model.__class__): model.state_dict() for model in models}

    DATA_MANAGER.save_python_obj(save_dict, f"{RESULTS_DIR}/{DATA_MANAGER.stamp}/{MODELS_DIR}/{suffix}")


# needed to load in class references
_read_all_class_names()
