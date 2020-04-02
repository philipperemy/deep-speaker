#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pickle

import click

from audio import Audio
from batcher import generate_cache_from_training_inputs, KerasConverter
from constants import SAMPLE_RATE
from train_cli import start_training
from utils import ClickType as Ct
from utils import init_pandas, create_new_empty_dir

logger = logging.getLogger(__name__)

VERSION = '1.0a'


@click.group()
def cli():
    logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s', level=logging.INFO)
    init_pandas()


@cli.command('version', short_help='Prints the version.')
def version():
    print(f'Version is {VERSION}.')


@cli.command('build-audio-cache', short_help='Build audio cache.')
@click.option('--audio_dir', required=True, type=Ct.input_dir())
@click.option('--cache_dir', required=True, type=Ct.output_dir())
@click.option('--sample_rate', default=SAMPLE_RATE, show_default=True, type=int)
@click.option('--parallel/--no-parallel', default=False, show_default=True)
def build_audio_cache(audio_dir, cache_dir, sample_rate, parallel):
    create_new_empty_dir(cache_dir)
    audio_reader = Audio(
        input_audio_dir=audio_dir,
        output_cache_dir=cache_dir,
        sample_rate=sample_rate,
        multi_threading=parallel
    )
    audio_reader.build_cache()


@cli.command('build-inputs-cache', short_help='Build model inputs cache.')
@click.option('--audio_dir', required=True, type=Ct.input_dir())
@click.option('--cache_dir', required=True, type=Ct.input_dir())
@click.option('--sample_rate', default=SAMPLE_RATE, show_default=True, type=int)
@click.option('--parallel/--no-parallel', default=False, show_default=True)
def build_inputs_cache(audio_dir, cache_dir, sample_rate, parallel):
    audio_reader = Audio(
        input_audio_dir=audio_dir,
        output_cache_dir=cache_dir,
        sample_rate=sample_rate,
        multi_threading=parallel
    )
    generate_cache_from_training_inputs(
        cache_dir,
        audio_reader,
        parallel
    )


@cli.command('build-keras-inputs', short_help='Build inputs to Keras.')
@click.option('--data_filename', required=True, type=Ct.input_file())
@click.option('--cache_dir', required=True, type=Ct.input_dir())
def build_keras_inputs(data_filename, cache_dir):
    with open(data_filename, 'rb') as r:
        data = pickle.load(r)
    kc = KerasConverter(cache_dir)
    kc.load_from_disk()
    kc.convert(data)
    kc.persist_from_disk()


@cli.command('train-model', short_help='Train a Keras model.')
@click.option('--cache_dir', required=True, type=Ct.input_dir())
@click.option('--checkpoints_dir', default='checkpoints', show_default=True, type=Ct.output_dir())
def build_keras_inputs(cache_dir, checkpoints_dir):
    kc = KerasConverter(cache_dir)
    kc.load_from_disk()
    start_training(checkpoints_dir, kc)


if __name__ == '__main__':
    cli()
