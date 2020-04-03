#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import click

from audio import Audio
from batcher import KerasConverter, FBankProcessor
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
@click.option('--working_dir', required=True, type=Ct.output_dir())
@click.option('--sample_rate', default=SAMPLE_RATE, show_default=True, type=int)
@click.option('--parallel/--no-parallel', default=False, show_default=True)
def build_audio_cache(audio_dir, working_dir, sample_rate, parallel):
    create_new_empty_dir(working_dir)
    audio_reader = Audio(
        input_audio_dir=audio_dir,
        output_working_dir=working_dir,
        sample_rate=sample_rate,
        multi_threading=parallel
    )
    audio_reader.build_cache()


@cli.command('build-inputs-cache', short_help='Build model inputs cache.')
@click.option('--audio_dir', required=True, type=Ct.input_dir())
@click.option('--working_dir', required=True, type=Ct.input_dir())
@click.option('--sample_rate', default=SAMPLE_RATE, show_default=True, type=int)
@click.option('--parallel/--no-parallel', default=False, show_default=True)
def build_inputs_cache(audio_dir, working_dir, sample_rate, parallel):
    audio_reader = Audio(
        input_audio_dir=audio_dir,
        output_working_dir=working_dir,
        sample_rate=sample_rate,
        multi_threading=parallel
    )
    inputs_generator = FBankProcessor(
        working_dir=working_dir,
        audio_reader=audio_reader,
        count_per_speaker=3000,
        speakers_sub_list=None,
        parallel=parallel
    )
    inputs_generator.generate()


@cli.command('build-keras-inputs', short_help='Build inputs to Keras.')
@click.option('--working_dir', required=True, type=Ct.input_dir())
def build_keras_inputs(working_dir):
    kc = KerasConverter(working_dir)
    kc.convert()
    kc.persist_to_disk()


@cli.command('train-model', short_help='Train a Keras model.')
@click.option('--working_dir', required=True, type=Ct.input_dir())
@click.option('--loss_on_softmax/--no_loss_on_softmax', default=True, show_default=True)
@click.option('--loss_on_embeddings/--no_loss_on_embeddings', default=False, show_default=True)
@click.option('--normalize_embeddings/--normalize_embeddings', default=False, show_default=True)
def train_model(working_dir, loss_on_softmax, loss_on_embeddings, normalize_embeddings):
    # 1/ --loss_on_softmax
    # 2/ --loss_on_embeddings --normalize_embeddings
    kc = KerasConverter(working_dir)
    start_training(kc, loss_on_softmax, loss_on_embeddings, normalize_embeddings)


if __name__ == '__main__':
    cli()
