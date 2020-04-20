import logging
import os
import random
import shutil
from collections import Counter
from glob import glob
from pathlib import Path

import click
import dill
import numpy as np
import pandas as pd
from natsort import natsorted

logger = logging.getLogger(__name__)


def find_files(directory, ext='wav'):
    return sorted(glob(directory + f'/**/*.{ext}', recursive=True))


def init_pandas():
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)


def create_new_empty_dir(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def ensure_dir_for_filename(filename: str):
    ensures_dir(os.path.dirname(filename))


def ensures_dir(directory: str):
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)


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


def load_best_checkpoint(checkpoint_dir):
    checkpoints = natsorted(glob(os.path.join(checkpoint_dir, '*.h5')))
    if len(checkpoints) != 0:
        return checkpoints[-1]
    return None


def delete_older_checkpoints(checkpoint_dir, max_to_keep=5):
    assert max_to_keep > 0
    checkpoints = natsorted(glob(os.path.join(checkpoint_dir, '*.h5')))
    checkpoints_to_keep = checkpoints[-max_to_keep:]
    for checkpoint in checkpoints:
        if checkpoint not in checkpoints_to_keep:
            os.remove(checkpoint)


def enable_deterministic():
    print('Deterministic mode enabled.')
    np.random.seed(123)
    random.seed(123)


def load_pickle(file):
    if not os.path.exists(file):
        return None
    logger.info(f'Loading PKL file: {file}.')
    with open(file, 'rb') as r:
        return dill.load(r)


def load_npy(file):
    if not os.path.exists(file):
        return None
    logger.info(f'Loading NPY file: {file}.')
    return np.load(file)


def libri_to_vctk_format(libri, subset, output):
    # INPUT: LibriSpeech/subset/speaker/utterance/utterance.wav
    # OUTPUT: Dataset/speaker/speaker_utterance.wav
    create_new_empty_dir(output)
    subsets = [p for p in Path(libri).iterdir() if p.is_dir()]
    speaker_counter = Counter()
    for s in subsets:
        if subset is not None and s.name != subset:
            continue
        for speaker in [a.name for a in s.iterdir()]:
            output_speaker_dir = os.path.join(output, speaker)
            ensures_dir(output_speaker_dir)
            speaker_wav_files = find_files(str(s / speaker), ext='flac')
            for speaker_wav_file in speaker_wav_files:
                output_filename = os.path.join(output_speaker_dir, f'{speaker}_{speaker_counter[speaker]}.wav')
                shutil.copy(speaker_wav_file, output_filename)
                speaker_counter[speaker] += 1
            logger.info(f'Speaker: {speaker}, {len(speaker_wav_files)} utterances. Processed...')
