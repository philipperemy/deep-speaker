import logging
import os
import pickle

import numpy as np

from constants import c
from speech_features import get_mfcc_features_390

logger = logging.getLogger(__name__)


def data_to_keras(data):
    categorical_speakers = SpeakersToCategorical(data)
    kx_train, ky_train, kx_test, ky_test = [], [], [], []
    ky_test = []
    for speaker_id in categorical_speakers.get_speaker_ids():
        d = data[speaker_id]
        y = categorical_speakers.get_one_hot_vector(d['speaker_id'])
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

    return kx_train, ky_train, kx_test, ky_test, categorical_speakers


def generate_features(audio_entities, max_count):
    features = []
    for _ in range(max_count):
        audio_entity = np.random.choice(audio_entities)
        voice_only_signal = audio_entity['audio_voice_only']
        cuts = np.random.uniform(low=1, high=len(voice_only_signal), size=2)
        signal_to_process = voice_only_signal[int(min(cuts)):int(max(cuts))]
        features_per_conv = get_mfcc_features_390(signal_to_process, c.AUDIO.SAMPLE_RATE, max_frames=None)
        if len(features_per_conv) > 0:
            features.append(features_per_conv)
    return features


def normalize(list_matrices, mean, std):
    return [(m - mean) / std for m in list_matrices]


class GenerateInputs:

    def __init__(self, cache_dir, max_count_per_class=500,
                 speakers_sub_list=None, multi_threading=False):
        self.cache_dir = cache_dir
        self.multi_threading = multi_threading
        self.inputs_dir = os.path.join(self.cache_dir, 'inputs')
        self.max_count_per_class = max_count_per_class
        if not os.path.exists(self.inputs_dir):
            os.makedirs(self.inputs_dir)
        from audio_reader import AudioReader, extract_speaker_id
        self.audio = AudioReader(audio_dir=c.AUDIO.VCTK_CORPUS_PATH,
                                 sample_rate=c.AUDIO.SAMPLE_RATE,
                                 cache_dir=c.AUDIO.CACHE_PATH,
                                 speakers_sub_list=speakers_sub_list)
        per_speaker_dict = {}
        for filename, audio_entity in self.audio.cache.items():
            speaker_id = extract_speaker_id(audio_entity['filename'])
            if speaker_id not in per_speaker_dict:
                per_speaker_dict[speaker_id] = []
            per_speaker_dict[speaker_id].append(audio_entity)
        self.per_speaker_dict = per_speaker_dict

    def start(self):
        if self.multi_threading:
            num_threads = os.cpu_count() // 2
            parallel_function(self.generate_inputs, sorted(self.per_speaker_dict), num_threads)
        else:
            for s in self.per_speaker_dict:
                self.generate_inputs(s)

    def generate_inputs(self, speaker_id):
        audio_entities = self.per_speaker_dict[speaker_id]
        logger.info('Processing speaker id = {}.'.format(speaker_id))
        cutoff = int(len(audio_entities) * 0.8)
        audio_entities_train = audio_entities[0:cutoff]
        audio_entities_test = audio_entities[cutoff:]

        train = generate_features(audio_entities_train, self.max_count_per_class)
        logger.info('Generated {} inputs for train/'.format(self.max_count_per_class))
        test = generate_features(audio_entities_test, self.max_count_per_class)
        logger.info('Generated {} inputs for test/'.format(self.max_count_per_class))

        mean_train = np.mean([np.mean(t) for t in train])
        std_train = np.mean([np.std(t) for t in train])

        train = normalize(train, mean_train, std_train)
        test = normalize(test, mean_train, std_train)

        if speaker_id in c.AUDIO.SPEAKERS_TRAINING_SET:
            output_filename = os.path.join(self.inputs_dir, speaker_id + '.pkl')
            inputs = {'train': train, 'test': test, 'speaker_id': speaker_id,
                      'mean_train': mean_train, 'std_train': std_train}
            with open(output_filename, 'wb') as w:
                pickle.dump(obj=inputs, file=w)
            logger.info('Adding speaker to the training dataset: {}'.format(speaker_id))
        else:
            logger.info('Discarding speaker for the training dataset: {}'.format(speaker_id))


class SpeakersToCategorical:
    def __init__(self, data):
        from keras.utils import to_categorical
        self.speaker_ids = sorted(list(data.keys()))
        self.int_speaker_ids = list(range(len(self.speaker_ids)))
        self.map_speakers_to_index = dict([(k, v) for (k, v) in zip(self.speaker_ids, self.int_speaker_ids)])
        self.map_index_to_speakers = dict([(v, k) for (k, v) in zip(self.speaker_ids, self.int_speaker_ids)])
        self.speaker_categories = to_categorical(self.int_speaker_ids, num_classes=len(self.speaker_ids))

    def get_speaker_from_index(self, index):
        return self.map_index_to_speakers[index]

    def get_one_hot_vector(self, speaker_id):
        index = self.map_speakers_to_index[speaker_id]
        return self.speaker_categories[index]

    def get_speaker_ids(self):
        return self.speaker_ids


def parallel_function(f, sequence, num_threads=None):
    from multiprocessing import Pool
    pool = Pool(processes=num_threads)
    result = pool.map(f, sequence)
    cleaned = [x for x in result if x is not None]
    pool.close()
    pool.join()
    return cleaned