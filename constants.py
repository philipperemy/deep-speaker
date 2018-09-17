import json
import logging
import os

import namedtupled

logger = logging.getLogger(__name__)

CONFIGURATION_FILENAME = 'conf.json'


def filename_to_named_tuple(filename):
    with open(filename) as data_file:
        c_ = json.load(data_file)
        # pprint(c_)
        return namedtupled.map(c_)


def load_constants():
    c_ = None
    try:
        c_ = filename_to_named_tuple(CONFIGURATION_FILENAME)
    except FileNotFoundError as e:
        try:
            c_ = filename_to_named_tuple(os.path.join('..', CONFIGURATION_FILENAME))
        except FileNotFoundError as e:
            try:
                c_ = filename_to_named_tuple(os.path.join('..', '..', CONFIGURATION_FILENAME))
            except:
                logger.error(e)
                logger.error('Please execute this command: cp conf.json.example conf.json')
    return c_


c = load_constants()
