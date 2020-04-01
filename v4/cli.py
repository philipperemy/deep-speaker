#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import click

from audio import AudioReader
from batcher import generate_cache_from_training_inputs
from constants import SAMPLE_RATE
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
    audio_reader = AudioReader(
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
    audio_reader = AudioReader(
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


if __name__ == '__main__':
    cli()
