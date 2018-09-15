import json
import logging
import os
import pickle
import shutil
import time
from argparse import ArgumentParser

from audio.audio_reader import AudioReader
from constants import c
from ml.utils import generate_inputs


def arg_parse():
    arg_p = ArgumentParser()
    arg_p.add_argument('--generate_audio_cache', action='store_true')
    arg_p.add_argument('--generate_training_inputs', action='store_true')
    return arg_p


def generate_cache_from_audio_files():
    cache_output_dir = os.path.expanduser(c.AUDIO.CACHE_PATH)
    print('The directory containing the cache is {}.'.format(cache_output_dir))
    print('Going to wipe out and regenerate the cache in 5 seconds. Ctrl+C to kill this script.')
    time.sleep(5)
    try:
        shutil.rmtree(cache_output_dir)
    except:
        pass
    os.makedirs(cache_output_dir)
    AudioReader(audio_dir=c.AUDIO.VCTK_CORPUS_PATH,
                sample_rate=c.AUDIO.SAMPLE_RATE,
                cache_dir=cache_output_dir,
                multi_threading_cache_generation=True,
                speakers_sub_list=None)


def generate_cache_from_training_inputs():
    cache_dir = os.path.expanduser(c.AUDIO.CACHE_PATH)
    data_filename = os.path.join(cache_dir, 'inputs-data.pkl')
    norm_filename = os.path.join(cache_dir, 'inputs-norm-constants.json')
    print('Data filename = {}'.format(data_filename))
    if not os.path.exists(data_filename):
        print('Data does not exist. Generating it now.')
        data, norm_data = generate_inputs(max_count_per_class=1000)
        pickle.dump(data, open(data_filename, 'wb'))
        json.dump(norm_data, fp=open(norm_filename, 'wb'), indent=4)
    else:
        print('Data found. No generation is necessary.')


def main():
    args = arg_parse().parse_args()

    if args.generate_audio_cache:
        generate_cache_from_audio_files()

    if args.generate_training_inputs:
        generate_cache_from_training_inputs()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
