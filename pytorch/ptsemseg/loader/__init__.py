import json
from ptsemseg.loader.cars import carsLoader

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
	    'cars': carsLoader,
    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']
