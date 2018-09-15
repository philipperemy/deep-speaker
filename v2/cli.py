import logging
import os
import shutil
import time
from argparse import ArgumentParser

from audio_reader import AudioReader
from constants import c
from utils import GenerateInputs

DEBUG_SPEAKERS_TRAINING = ['p225', 'p226']


def arg_parse():
    arg_p = ArgumentParser()
    arg_p.add_argument('--generate_audio_cache', action='store_true')
    arg_p.add_argument('--generate_training_inputs', action='store_true')
    arg_p.add_argument('--debug', action='store_true')
    return arg_p


def generate_cache_from_audio_files(args):
    cache_output_dir = os.path.expanduser(c.AUDIO.CACHE_PATH)
    print('The directory containing the cache is {}.'.format(cache_output_dir))
    print('Going to wipe out and regenerate the cache in 5 seconds. Ctrl+C to kill this script.')
    time.sleep(5)
    try:
        shutil.rmtree(cache_output_dir)
    except:
        pass
    os.makedirs(cache_output_dir)

    speakers_sub_list = None
    if args.debug:
        speakers_sub_list = DEBUG_SPEAKERS_TRAINING

    AudioReader(audio_dir=c.AUDIO.VCTK_CORPUS_PATH,
                sample_rate=c.AUDIO.SAMPLE_RATE,
                cache_dir=cache_output_dir,
                multi_threading_cache_generation=True,
                speakers_sub_list=speakers_sub_list)


def generate_cache_from_training_inputs(args):
    cache_dir = os.path.expanduser(c.AUDIO.CACHE_PATH)
    speakers_sub_list = None
    if args.debug:
        speakers_sub_list = DEBUG_SPEAKERS_TRAINING
    GenerateInputs(cache_dir, max_count_per_class=1000, speakers_sub_list=speakers_sub_list).start()


def main():
    args = arg_parse().parse_args()

    if args.generate_audio_cache:
        generate_cache_from_audio_files(args)
        exit(1)

    if args.generate_training_inputs:
        generate_cache_from_training_inputs(args)
        exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
