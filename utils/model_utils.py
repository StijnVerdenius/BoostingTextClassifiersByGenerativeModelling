import importlib
import inspect
from typing import List

import torch.nn as nn
import torch.optim as opt

from utils.constants import *

types = [CLASS_DIR, GEN_DIR, LOSS_DIR, DATASETS]
models = {x: {} for x in types}


def _read_all_class_names():
    """
    private function that imports all class references in a dictionary
    """

    for typ in types:
        for name in os.listdir(os.path.join(".", "models", typ)):
            if not "__" in name:
                short_name = str(name.split(".")[0])
                short_name: str
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


def detach_list(items):
    for i, x in enumerate(items):
        items[i] = x.detach()


def delete_list(items):
    for i, x in enumerate(items):
        del items[i], x


def save_models(models: List[nn.Module],
                suffix: str):
    """
    Saves current state of models
    """

    save_dict = {str(model.__class__): model.state_dict() for model in models}

    DATA_MANAGER.save_python_obj(save_dict, os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, MODELS_DIR, suffix))


# needed to load in class references
_read_all_class_names()


def calculate_accuracy(targets, output, *ignored):
    probabilities = output.log_softmax(dim=-1)
    _, classifications = probabilities.detach().max(dim=-1)
    return (targets.eq(classifications)).float().mean()


def assert_type(content, expected_type):
    """ makes sure type is respected"""

    assert_non_empty(content)
    func = inspect.stack()[1][3]
    assert isinstance(content, type(expected_type)), "No {} entered in {} but instead value {}".format(str(expected_type),
                                                                                                 func,
                                                                                                 str(content))


def assert_non_empty(content):
    """ makes sure not None or len()==0 """

    func = inspect.stack()[1][3]
    assert not content == None, "Content is null in {}".format(func)
    if type(content) is list or type(content) == str:
        assert len(content) > 0, "Empty {} in {}".format(type(content), func)
