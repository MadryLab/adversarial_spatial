from collections import namedtuple
import json

import numpy as np

def get_config(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)

    return config


def config_to_namedtuple(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = config_to_namedtuple(value) 
        return namedtuple('GenericDict', obj.keys())(**obj)
    elif isinstance(obj, list):
        return [config_to_namedtuple(item) for item in obj]
    else:
        return obj
