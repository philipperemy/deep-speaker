import logging
import os
import pickle
from collections import defaultdict
from random import choice

import dill
import numpy as np
from keras.utils import to_categorical
from python_speech_features import fbank, delta

from audio import extract_speaker_id
from constants import SAMPLE_RATE, TRAIN_TEST_RATIO
from utils import parallel_function, ensures_dir, find_files

logger = logging.getLogger(__name__)


def pre_process_inputs(sig=np.random.uniform(size=32000), target_sample_rate=8000):
    filter_banks, energies = fbank(sig, samplerate=target_sample_rate, nfilt=64)
    delta_1 = delta(filter_banks, N=1)
    delta_2 = delta(delta_1, N=1)
    # (num_frames, n_filters, 3).
    frames_features = np.transpose(np.stack([filter_banks, delta_1, delta_2]), (1, 2, 0))
    return frames_features


def generate_features(audio_entities, max_count):
    features = []
    for _ in range(max_count):
        audio_entity = np.random.choice(audio_entities)
        voice_only_signal = audio_entity['audio_voice_only']
        cut = choice(range(SAMPLE_RATE // 10))
        signal_to_process = voice_only_signal[cut:]
        features.append(pre_process_inputs(signal_to_process, SAMPLE_RATE))
    return np.array(features)


def normalize(list_matrices, mean, std):
    return [(m - mean) / std for m in list_matrices]


class KerasConverter:

    def __init__(self, saved_dir):
        self.saved_dir = saved_dir
        self.output_dir = os.path.join(self.saved_dir, 'keras-converter')
        ensures_dir(self.output_dir)
        self.categorical_speakers = None
        self.kx_train = None
        self.kx_test = None
        self.ky_train = None
        self.ky_test = None

    def load_from_disk(self):

        def load(file):
            if not os.path.exists(file):
                return None
            with open(file, 'rb') as r:
                return pickle.load(r)

        self.categorical_speakers = load(os.path.join(self.output_dir, 'categorical_speakers.pkl'))
        self.kx_train = load(os.path.join(self.output_dir, 'kx_train.pkl'))
        self.kx_test = load(os.path.join(self.output_dir, 'kx_test.pkl'))
        self.ky_train = load(os.path.join(self.output_dir, 'ky_train.pkl'))
        self.ky_test = load(os.path.join(self.output_dir, 'ky_test.pkl'))

    def persist_from_disk(self):
        with open(os.path.join(self.output_dir, 'categorical_speakers.pkl'), 'wb') as w:
            pickle.dump(self.categorical_speakers, w)
        with open(os.path.join(self.output_dir, 'kx_train.pkl'), 'wb') as w:
            pickle.dump(self.kx_train, w)
        with open(os.path.join(self.output_dir, 'kx_test.pkl'), 'wb') as w:
            pickle.dump(self.kx_test, w)
        with open(os.path.join(self.output_dir, 'ky_train.pkl'), 'wb') as w:
            pickle.dump(self.ky_train, w)
        with open(os.path.join(self.output_dir, 'ky_test.pkl'), 'wb') as w:
            pickle.dump(self.ky_test, w)

    def convert(self, data):
        categorical_speakers = OneHotSpeakers(data)
        kx_train, ky_train, kx_test, ky_test = [], [], [], []
        ky_test = []
        for speaker_id in categorical_speakers.get_speaker_ids():
            d = data[speaker_id]
            y = categorical_speakers.get_one_hot(d['speaker_id'])
            for x_train_elt in data[speaker_id]['train']:
                for x_train_sub_elt in x_train_elt:
                    kx_train.append(x_train_sub_elt)
                    ky_train.append(y)

            for x_test_elt in data[speaker_id]['test']:
                for x_test_sub_elt in x_test_elt:
                    kx_test.append(x_test_sub_elt)
                    ky_test.append(y)

        kx_train = np.array(kx_train)
        kx_test = np.array(kx_test)

        ky_train = np.array(ky_train)
        ky_test = np.array(ky_test)

        self.categorical_speakers = categorical_speakers
        self.kx_train, self.ky_train, self.kx_test, self.ky_test = kx_train, ky_train, kx_test, ky_test

        return kx_train, ky_train, kx_test, ky_test, categorical_speakers


class InputsGenerator:

    def __init__(self, cache_dir, audio_reader, max_count_per_class=50,
                 speakers_sub_list=None, parallel=False):
        self.cache_dir = os.path.expanduser(cache_dir)
        self.audio_reader = audio_reader
        self.parallel = parallel
        self.model_inputs_dir = os.path.join(self.cache_dir, 'inputs')
        self.max_count_per_class = max_count_per_class
        ensures_dir(self.model_inputs_dir)
        self.speaker_ids = self.audio_reader.all_speaker_ids if speakers_sub_list is None else speakers_sub_list

    def start_generation(self):
        logger.info('Starting the inputs generation...')
        if self.parallel:
            num_threads = os.cpu_count()
            logger.info(f'Using {num_threads} threads.')
            parallel_function(self.generate_and_dump_inputs_to_pkl, sorted(self.speaker_ids), num_threads)
        else:
            logger.info('Using only 1 thread.')
            for s in self.speaker_ids:
                self.generate_and_dump_inputs_to_pkl(s)

        logger.info('Generating the unified inputs pkl file.')
        full_inputs = {}
        for inputs_filename in find_files(self.model_inputs_dir, 'pkl'):
            with open(inputs_filename, 'rb') as r:
                inputs = pickle.load(r)
                logger.info(f'Read {inputs_filename}.')
            full_inputs[inputs['speaker_id']] = inputs
        full_inputs_output_filename = os.path.join(self.cache_dir, 'full_inputs.pkl')
        # dill can manage with files larger than 4GB.
        with open(full_inputs_output_filename, 'wb') as w:
            dill.dump(obj=full_inputs, file=w)
        logger.info(f'[DUMP UNIFIED INPUTS] {full_inputs_output_filename}.')

    def generate_and_dump_inputs_to_pkl(self, speaker_id):
        output_filename = os.path.join(self.model_inputs_dir, speaker_id + '.pkl')
        if os.path.isfile(output_filename):
            logger.info(f'Inputs file already exists: {output_filename}.')
            return

        inputs = self.generate_inputs(speaker_id)
        with open(output_filename, 'wb') as w:
            pickle.dump(obj=inputs, file=w)
        logger.info(f'[DUMP INPUTS] {output_filename}')

    def generate_inputs_for_inference(self, speaker_id):
        speaker_cache, metadata = self.audio_reader.load_cache([speaker_id])
        audio_entities = list(speaker_cache.values())
        logger.info(f'Generating the inputs necessary for the inference (speaker is {speaker_id})...')
        logger.info('This might take a couple of minutes to complete.')
        feat = generate_features(audio_entities, self.max_count_per_class)
        mean = np.mean([np.mean(t) for t in feat])
        std = np.mean([np.std(t) for t in feat])
        feat = normalize(feat, mean, std)
        return feat

    def generate_inputs(self, speaker_id):
        per_speaker_dict = defaultdict(list)
        cache, metadata = self.audio_reader.load_cache([speaker_id])

        for filename, audio_entity in cache.items():
            speaker_id_2 = extract_speaker_id(audio_entity['filename'])
            assert speaker_id_2 == speaker_id, f'{speaker_id} {speaker_id_2}'
            per_speaker_dict[speaker_id].append(audio_entity)

        audio_entities = per_speaker_dict[speaker_id]
        cutoff = int(len(audio_entities) * TRAIN_TEST_RATIO)
        audio_entities_train = audio_entities[0:cutoff]
        audio_entities_test = audio_entities[cutoff:]

        train = generate_features(audio_entities_train, self.max_count_per_class)
        test = generate_features(audio_entities_test, self.max_count_per_class)
        logger.info(f'Generated {self.max_count_per_class}/{self.max_count_per_class} '
                    f'inputs for train/test for speaker {speaker_id}.')

        # TODO: check that.
        mean_train = np.mean([np.mean(t) for t in train])
        std_train = np.mean([np.std(t) for t in train])

        train = normalize(train, mean_train, std_train)
        test = normalize(test, mean_train, std_train)

        return {
            'train': train,
            'test': test,
            'speaker_id': speaker_id,
            'mean_train': mean_train,
            'std_train': std_train
        }


class OneHotSpeakers:

    def __init__(self, data):
        self.speaker_ids = sorted(list(data.keys()))
        self.int_speaker_ids = list(range(len(self.speaker_ids)))
        self.map_speakers_to_index = dict([(k, v) for (k, v) in zip(self.speaker_ids, self.int_speaker_ids)])
        self.map_index_to_speakers = dict([(v, k) for (k, v) in zip(self.speaker_ids, self.int_speaker_ids)])
        self.speaker_categories = to_categorical(self.int_speaker_ids, num_classes=len(self.speaker_ids))

    def get_speaker_from_index(self, index):
        return self.map_index_to_speakers[index]

    def get_one_hot(self, speaker_id):
        index = self.map_speakers_to_index[speaker_id]
        return self.speaker_categories[index]

    def get_speaker_ids(self):
        return self.speaker_ids
