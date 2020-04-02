import logging
import os
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


def mfcc_fbank(sig=np.random.uniform(size=32000), target_sample_rate=8000):
    # Returns MFCC with shape (num_frames, n_filters, 3).
    filter_banks, energies = fbank(sig, samplerate=target_sample_rate, nfilt=64)
    delta_1 = delta(filter_banks, N=1)
    delta_2 = delta(delta_1, N=1)
    frames_features = np.transpose(np.stack([filter_banks, delta_1, delta_2]), (1, 2, 0))
    return frames_features


def features(audio_entities, n):
    feat = []
    for _ in range(n):
        audio_entity = np.random.choice(audio_entities)
        voice_only_signal = audio_entity['audio_voice_only']
        cut = choice(range(SAMPLE_RATE // 10))
        signal_to_process = voice_only_signal[cut:]
        feat.append(mfcc_fbank(signal_to_process, SAMPLE_RATE))
    return np.array(feat)


def normalize(list_matrices, mean, std):
    return [(m - mean) / std for m in list_matrices]


class KerasConverter:

    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.data_filename = os.path.join(self.working_dir, 'complete_fbank_inputs.pkl')
        self.output_dir = os.path.join(self.working_dir, 'keras-inputs')
        with open(self.data_filename, 'rb') as r:
            self.data = dill.load(r)
        ensures_dir(self.output_dir)
        self.categorical_speakers = None
        self.kx_train = None
        self.kx_test = None
        self.ky_train = None
        self.ky_test = None
        self.load_from_disk()

    def load_from_disk(self):

        def load(file):
            if not os.path.exists(file):
                return None
            with open(file, 'rb') as r:
                return dill.load(r)

        self.categorical_speakers = load(os.path.join(self.output_dir, 'categorical_speakers.pkl'))
        self.kx_train = load(os.path.join(self.output_dir, 'kx_train.pkl'))
        self.kx_test = load(os.path.join(self.output_dir, 'kx_test.pkl'))
        self.ky_train = load(os.path.join(self.output_dir, 'ky_train.pkl'))
        self.ky_test = load(os.path.join(self.output_dir, 'ky_test.pkl'))

    def persist_to_disk(self):
        with open(os.path.join(self.output_dir, 'categorical_speakers.pkl'), 'wb') as w:
            dill.dump(self.categorical_speakers, w)
        with open(os.path.join(self.output_dir, 'kx_train.pkl'), 'wb') as w:
            dill.dump(self.kx_train, w)
        with open(os.path.join(self.output_dir, 'kx_test.pkl'), 'wb') as w:
            dill.dump(self.kx_test, w)
        with open(os.path.join(self.output_dir, 'ky_train.pkl'), 'wb') as w:
            dill.dump(self.ky_train, w)
        with open(os.path.join(self.output_dir, 'ky_test.pkl'), 'wb') as w:
            dill.dump(self.ky_test, w)

    def convert(self, max_length=28):  # say xx fbank frames for now bitch.
        categorical_speakers = OneHotSpeakers(self.data)
        kx_train, ky_train, kx_test, ky_test = [], [], [], []
        ky_test = []
        for speaker_id in categorical_speakers.get_speaker_ids():
            d = self.data[speaker_id]
            y = categorical_speakers.get_one_hot(d['speaker_id'])
            for x_train_elt in self.data[speaker_id]['train']:
                kx_train.append(x_train_elt[0:max_length])
                ky_train.append(y)

            for x_test_elt in self.data[speaker_id]['test']:
                kx_test.append(x_test_elt[0:max_length])
                ky_test.append(y)

        kx_train = np.array(kx_train)
        print(f'kx_train.shape = {kx_train.shape}')
        kx_test = np.array(kx_test)
        print(f'kx_test.shape = {kx_test.shape}')
        ky_train = np.array(ky_train)
        print(f'ky_train.shape = {ky_train.shape}')
        ky_test = np.array(ky_test)
        print(f'ky_test.shape = {ky_test.shape}')

        self.categorical_speakers = categorical_speakers
        self.kx_train, self.ky_train, self.kx_test, self.ky_test = kx_train, ky_train, kx_test, ky_test

        return kx_train, ky_train, kx_test, ky_test, categorical_speakers


class FBankProcessor:

    def __init__(self, working_dir, audio_reader, count_per_speaker=50,
                 speakers_sub_list=None, parallel=False):
        self.working_dir = os.path.expanduser(working_dir)
        self.audio_reader = audio_reader
        self.parallel = parallel
        self.model_inputs_dir = os.path.join(self.working_dir, 'fbank-inputs')
        self.count_per_speaker = count_per_speaker
        ensures_dir(self.model_inputs_dir)
        self.speaker_ids = self.audio_reader.all_speaker_ids if speakers_sub_list is None else speakers_sub_list

    def generate(self):
        if self.parallel:
            num_proc = os.cpu_count()
            logger.info(f'Using {num_proc} threads.')
            parallel_function(self._cache_inputs, sorted(self.speaker_ids), num_proc)
        else:
            logger.info('Using only 1 thread.')
            for s in self.speaker_ids:
                self._cache_inputs(s)
        logger.info('Generating the unified inputs pkl file.')
        full_inputs = {}
        for inputs_filename in find_files(self.model_inputs_dir, ext='pkl'):
            with open(inputs_filename, 'rb') as r:
                inputs = dill.load(r)
                logger.info(f'Read {inputs_filename}.')
            full_inputs[inputs['speaker_id']] = inputs
        full_inputs_output_filename = os.path.join(self.working_dir, 'complete_fbank_inputs.pkl')
        # dill can manage with files larger than 4GB.
        with open(full_inputs_output_filename, 'wb') as w:
            dill.dump(obj=full_inputs, file=w)
        logger.info(f'[DUMP UNIFIED INPUTS] {full_inputs_output_filename}.')

    def _cache_inputs(self, speaker_id):
        output_filename = os.path.join(self.model_inputs_dir, speaker_id + '.pkl')
        if os.path.isfile(output_filename):
            logger.info(f'Inputs file already exists: {output_filename}.')
            return
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

        train = features(audio_entities_train, self.count_per_speaker)
        test = features(audio_entities_test, self.count_per_speaker)
        logger.info(f'Generated {self.count_per_speaker}/{self.count_per_speaker} '
                    f'fbank inputs (train/test) for speaker {speaker_id}.')

        # TODO: check that.
        mean_train = np.mean([np.mean(t) for t in train])
        std_train = np.mean([np.std(t) for t in train])

        train = normalize(train, mean_train, std_train)
        test = normalize(test, mean_train, std_train)

        inputs = {
            'train': train,
            'test': test,
            'speaker_id': speaker_id,
            'mean_train': mean_train,
            'std_train': std_train
        }
        with open(output_filename, 'wb') as w:
            dill.dump(obj=inputs, file=w)
        logger.info(f'[DUMP INPUTS] {output_filename}')

    def generate_inputs_for_inference(self, speaker_id):
        speaker_cache, metadata = self.audio_reader.load_cache([speaker_id])
        audio_entities = list(speaker_cache.values())
        logger.info(f'Generating the inputs necessary for the inference (speaker is {speaker_id})...')
        logger.info('This might take a couple of minutes to complete.')
        feat = features(audio_entities, self.count_per_speaker)
        mean = np.mean([np.mean(t) for t in feat])
        std = np.mean([np.std(t) for t in feat])
        feat = normalize(feat, mean, std)
        return feat


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
