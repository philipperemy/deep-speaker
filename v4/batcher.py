import logging
import os
from collections import defaultdict
from random import choice

import dill
import numpy as np
from keras.utils import to_categorical
from python_speech_features import fbank, delta
from tqdm import tqdm

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
    count = 0
    while count < n:
        try:
            audio_entity = np.random.choice(audio_entities)
            voice_only_signal = audio_entity['audio_voice_only']
            cut = choice(range(SAMPLE_RATE // 10))
            signal_to_process = voice_only_signal[cut:]
            feat.append(mfcc_fbank(signal_to_process, SAMPLE_RATE))
            count += 1
        except IndexError:  # happens if signal is too small.
            pass
    return np.array(feat)


def normalize(list_matrices, mean, std):
    return [(m - mean) / std for m in list_matrices]


class KerasConverter:

    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.output_dir = os.path.join(self.working_dir, 'keras-inputs')
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

        def load2(file):
            if not os.path.exists(file):
                return None
            return np.load(file)

        self.categorical_speakers = load(os.path.join(self.output_dir, 'categorical_speakers.pkl'))
        self.kx_train = load2(os.path.join(self.output_dir, 'kx_train.npy'))
        self.kx_test = load2(os.path.join(self.output_dir, 'kx_test.npy'))
        self.ky_train = load2(os.path.join(self.output_dir, 'ky_train.npy'))
        self.ky_test = load2(os.path.join(self.output_dir, 'ky_test.npy'))

    def persist_to_disk(self):
        with open(os.path.join(self.output_dir, 'categorical_speakers.pkl'), 'wb') as w:
            dill.dump(self.categorical_speakers, w)
        np.save(os.path.join(self.output_dir, 'kx_train.npy'), self.kx_train)
        np.save(os.path.join(self.output_dir, 'kx_test.npy'), self.kx_test)
        np.save(os.path.join(self.output_dir, 'ky_train.npy'), self.ky_train)
        np.save(os.path.join(self.output_dir, 'ky_test.npy'), self.ky_test)

    def convert(self, max_length=28):  # TODO: say xx fbank frames for now bitch.
        fbank_files = os.path.join(self.working_dir, 'fbank-inputs')
        speakers_list = [os.path.splitext(os.path.basename(a))[0] for a in find_files(fbank_files, ext='pkl')]
        categorical_speakers = OneHotSpeakers(speakers_list)

        # num_samples = len(speakers_list) *
        kx_train, ky_train, kx_test, ky_test = None, None, None, None
        for speaker_id in tqdm(categorical_speakers.get_speaker_ids(), desc='Converting to Keras format'):
            with open(os.path.join(self.working_dir, 'fbank-inputs', speaker_id + '.pkl'), 'rb') as r:
                d = dill.load(r)
            y = categorical_speakers.get_one_hot(d['speaker_id'])

            if kx_train is None:
                num_samples_train = len(speakers_list) * len(d['train'])
                num_samples_test = len(speakers_list) * len(d['test'])

                # 64 fbanks 3 channels.
                # float32
                kx_train = np.zeros((num_samples_train, max_length, 64, 3), dtype=np.float32)
                ky_train = np.zeros((num_samples_train, len(speakers_list)), dtype=np.float32)

                kx_test = np.zeros((num_samples_test, max_length, 64, 3), dtype=np.float32)
                ky_test = np.zeros((num_samples_test, len(speakers_list)), dtype=np.float32)

                print(f'kx_train.shape = {kx_train.shape}')
                print(f'kx_test.shape = {kx_test.shape}')
                print(f'ky_train.shape = {ky_train.shape}')
                print(f'ky_test.shape = {ky_test.shape}')

            for i, x_train_elt in enumerate(d['train']):
                if len(x_train_elt) >= max_length:
                    st = choice(range(0, len(x_train_elt) - max_length + 1))
                    kx_train[i] = x_train_elt[st:st + max_length]
                    ky_train[i] = y
                else:
                    # simple for now.
                    kx_train[i] = kx_train[i - 1]
                    ky_train[i] = ky_train[i - 1]

            for i, x_test_elt in enumerate(d['test']):
                if len(x_test_elt) >= max_length:
                    st = choice(range(0, len(x_test_elt) - max_length + 1))
                    kx_test[i] = x_test_elt[st:st + max_length]
                    ky_test[i] = y
                else:
                    kx_test[i] = kx_test[i - 1]
                    ky_test[i] = ky_test[i - 1]

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

    def __init__(self, speakers_list):
        self.speaker_ids = sorted(speakers_list)
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
