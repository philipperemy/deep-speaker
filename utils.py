import os
import re
import shutil
from glob import glob


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_last_checkpoint_if_any(checkpoint_folder):
    files = glob('{}/*.h5'.format(checkpoint_folder), recursive=True)
    if len(files) == 0:
        return None
    return natural_sort(files)[-1]


def create_dir_and_delete_content(directory):
    try:
        shutil.rmtree(directory)
    except:
        pass
    os.makedirs(directory)
