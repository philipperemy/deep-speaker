import logging
import os
import random
import re
import shutil
from glob import glob

import click
import pandas as pd

logger = logging.getLogger(__name__)


def find_files(directory, extension='wav'):
    return sorted(glob(directory + f'/**/*.{extension}', recursive=True))


def init_pandas():
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)


def natural_sort(lst: list):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(lst, key=alphanum_key)


def create_new_empty_dir(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def ensure_dir_for_filename(filename: str):
    ensures_dir(os.path.dirname(filename))


def ensures_dir(directory: str):
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)


def split(arr, size):
    arrays = []
    while len(arr) > size:
        slice_ = arr[:size]
        arrays.append(slice_)
        arr = arr[size:]
    arrays.append(arr)
    return arrays


def shuffle(lst: list):
    random.seed(123)
    random.shuffle(lst)


class ClickType:

    @staticmethod
    def input_file(writable=False):
        return click.Path(exists=True, file_okay=True, dir_okay=False,
                          writable=writable, readable=True, resolve_path=True)

    @staticmethod
    def input_dir(writable=False):
        return click.Path(exists=True, file_okay=False, dir_okay=True,
                          writable=writable, readable=True, resolve_path=True)

    @staticmethod
    def output_file():
        return click.Path(exists=False, file_okay=True, dir_okay=False,
                          writable=True, readable=True, resolve_path=True)

    @staticmethod
    def output_dir():
        return click.Path(exists=False, file_okay=False, dir_okay=True,
                          writable=True, readable=True, resolve_path=True)


def parallel_function(f, sequence, num_threads=None):
    from multiprocessing import Pool
    pool = Pool(processes=num_threads)
    result = pool.map(f, sequence)
    cleaned = [x for x in result if x is not None]
    pool.close()
    pool.join()
    return cleaned
