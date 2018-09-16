import logging
import os
import shutil
import time
from argparse import ArgumentParser

from audio_reader import AudioReader
from constants import c
from utils import InputsGenerator

DEBUG_SPEAKERS_TRAINING = ['p225', 'p226']


def arg_parse():
    arg_p = ArgumentParser()
    arg_p.add_argument('--force_complete_audio_cache_regeneration', action='store_true')
    arg_p.add_argument('--generate_audio_cache', action='store_true')
    arg_p.add_argument('--generate_training_inputs', action='store_true')
    arg_p.add_argument('--debug', action='store_true')
    arg_p.add_argument('--multi_threading', action='store_true')
    return arg_p


def force_complete_audio_cache_regeneration(audio_reader, args):
    cache_output_dir = os.path.expanduser(c.AUDIO.CACHE_PATH)
    print('The directory containing the cache is {}.'.format(cache_output_dir))
    print('Going to wipe out and regenerate the cache in 5 seconds. Ctrl+C to kill this script.')
    time.sleep(5)
    try:
        shutil.rmtree(cache_output_dir)
    except:
        pass
    os.makedirs(cache_output_dir)
    audio_reader.build_cache()


def generate_cache_from_training_inputs(audio_reader, args):
    cache_dir = os.path.expanduser(c.AUDIO.CACHE_PATH)
    speakers_sub_list = None
    if args.debug:
        speakers_sub_list = DEBUG_SPEAKERS_TRAINING
    inputs_generator = InputsGenerator(cache_dir=cache_dir,
                                       audio_reader=audio_reader,
                                       max_count_per_class=1000,
                                       speakers_sub_list=speakers_sub_list,
                                       multi_threading=args.multi_threading)
    inputs_generator.start_generation()


def main():
    args = arg_parse().parse_args()

    audio_reader = AudioReader(input_audio_dir=c.AUDIO.VCTK_CORPUS_PATH,
                               output_cache_dir=c.AUDIO.CACHE_PATH,
                               sample_rate=c.AUDIO.SAMPLE_RATE,
                               multi_threading=args.multi_threading)

    if args.force_complete_audio_cache_regeneration:
        force_complete_audio_cache_regeneration(audio_reader, args)
        exit(1)

    if args.generate_audio_cache:
        audio_reader.build_cache()
        exit(1)

    if args.generate_training_inputs:
        generate_cache_from_training_inputs(audio_reader, args)
        exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
