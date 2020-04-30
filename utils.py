import logging
import os
import random
import shutil
from glob import glob

import click
import dill
import numpy as np
import pandas as pd
from natsort import natsorted

from constants import TRAIN_TEST_RATIO

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


def train_test_sp_to_utt(audio, is_test):
    sp_to_utt = {}
    for speaker_id, utterances in audio.speakers_to_utterances.items():
        utterances_files = sorted(utterances.values())
        train_test_sep = int(len(utterances_files) * TRAIN_TEST_RATIO)
        sp_to_utt[speaker_id] = utterances_files[train_test_sep:] if is_test else utterances_files[:train_test_sep]
    return sp_to_utt


def embedding_fusion(embeddings_1: np.array, embeddings_2: np.array):
    assert len(embeddings_1.shape) == 2  # (batch_size, 512).
    assert embeddings_1.shape == embeddings_2.shape
    embeddings_sum = embeddings_1 + embeddings_2
    fusion = embeddings_sum / np.linalg.norm(embeddings_sum, ord=2, axis=1, keepdims=True)
    assert np.all((-1 <= fusion) & (fusion <= 1))
    assert np.all(abs(np.sum(fusion ** 2, axis=1) - 1) < 1e-6)
    return fusion

def score_fusion(scores_1: np.array, scores_2: np.array):
    def normalize_scores(m, epsilon=1e-12):
        return (m - np.mean(m)) / max(np.std(m), epsilon)

    # score has to be between -1 and 1.
    return np.tanh(np.sum(normalize_scores(np.stack((scores_1, scores_2), axis=2)), axis=2))


if __name__ == '__main__':
    score_fusion(np.ones((5, 100)), np.ones((5, 100)))

