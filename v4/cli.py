#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import shutil
from pathlib import Path

import click
from tqdm import tqdm

from audio import Audio
from batcher import KerasConverter, FBankProcessor
from constants import SAMPLE_RATE, NUM_FRAMES
from train2 import start_training
from utils import ClickType as Ct, ensures_dir, create_new_empty_dir, find_files
from utils import init_pandas

logger = logging.getLogger(__name__)

VERSION = '1.0b'


@click.group()
def cli():
    logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s', level=logging.INFO)
    init_pandas()


@cli.command('version', short_help='Prints the version.')
def version():
    print(f'Version is {VERSION}.')


@cli.command('build-audio-cache', short_help='Build audio cache.')
@click.option('--audio_dir', required=True, type=Ct.input_dir())
@click.option('--working_dir', required=True, type=Ct.output_dir())
@click.option('--sample_rate', default=SAMPLE_RATE, show_default=True, type=int)
@click.option('--parallel/--no-parallel', default=False, show_default=True)
def build_audio_cache(audio_dir, working_dir, sample_rate, parallel):
    ensures_dir(working_dir)
    audio_reader = Audio(
        input_audio_dir=audio_dir,
        output_working_dir=working_dir,
        sample_rate=sample_rate,
        multi_threading=parallel
    )
    audio_reader.build_cache()


@cli.command('build-mfcc-cache', short_help='Build model inputs cache.')
@click.option('--audio_dir', required=True, type=Ct.input_dir())
@click.option('--working_dir', required=True, type=Ct.input_dir())
@click.option('--sample_rate', default=SAMPLE_RATE, show_default=True, type=int)
def build_inputs_cache(audio_dir, working_dir, sample_rate):
    audio_reader = Audio(
        input_audio_dir=audio_dir,
        output_working_dir=working_dir,
        sample_rate=sample_rate,
        multi_threading=False
    )
    inputs_generator = FBankProcessor(
        working_dir=working_dir,
        audio_reader=audio_reader,
        speakers_sub_list=None,
        parallel=False
    )
    inputs_generator.generate()


@cli.command('build-keras-inputs', short_help='Build inputs to Keras.')
@click.option('--working_dir', required=True, type=Ct.input_dir())
def build_keras_inputs(working_dir):
    counts_per_speaker = (5000, 500)  # train, test (5000, 500) had good results.
    kc = KerasConverter(working_dir)
    kc.generate(max_length=NUM_FRAMES, counts_per_speaker=counts_per_speaker)
    kc.persist_to_disk()


@cli.command('train-model', short_help='Train a Keras model.')
@click.option('--working_dir', required=True, type=Ct.input_dir())
@click.option('--pre_training_phase/--no_pre_training_phase', default=False, show_default=True)
def train_model(working_dir, pre_training_phase):
    # Default parameters: 0.97 accuracy on test set with [--loss_on_softmax].
    # p225 p226 p227 p228 p229 p230 p231 p232 p233 p234 p236 p237 p238 p239
    # 1/ --loss_on_softmax
    # 2/ --loss_on_embeddings --normalize_embeddings
    # We can easily get:
    # 011230, train(emb, last 100) = 0.37317 test(emb, last 100) = 0.37739

    # (5000, 500) gives great results. (1000, 100) we stall at 0.965. Patience is only 10.
    # On all VCTK Corpus with LeNet, 0.98 without doing much.

    # (5000, 500) gives good results (for first try). Frame is 128.
    # With the complicated model, we get 0.96 accuracy (epoch 10) which is not too bad.
    # We obviously over-fit on the dataset.

    # Let's try with (5000, 500) and bigger frame 160, dropout 0.5.
    start_training(working_dir, pre_training_phase)


@cli.command('libri-to-vctk-format', short_help='Converts Libri dataset to VCTK.')
@click.option('--libri', required=True, type=Ct.input_dir())
@click.option('--output', required=True, type=Ct.output_dir())
def libri_to_vctk(libri, output):
    create_new_empty_dir(output)
    while True:
        sub_dirs = [d for d in Path(libri).iterdir() if d.is_dir()]
        if len(sub_dirs) > 1:
            break
        libri = Path(sub_dirs[0])
    logger.info(libri)
    for speaker in sorted(os.listdir(libri)):
        speaker_wav_files = find_files(os.path.join(libri, speaker))
        output_speaker_dir = os.path.join(output, speaker)
        ensures_dir(output_speaker_dir)
        for i, speaker_wav_file in tqdm(enumerate(speaker_wav_files), desc=f'Speaker {speaker}'):
            output_filename = os.path.join(output_speaker_dir, f'{speaker}_{i}.wav')
            # logger.info(f'{speaker_wav_file} -> {output_filename}.')
            shutil.copy(speaker_wav_file, output_filename)


if __name__ == '__main__':
    cli()
