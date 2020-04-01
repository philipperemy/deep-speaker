import logging
import os
import pickle
from collections import defaultdict

import dill
import numpy as np
import pandas as pd

from audio import extract_speaker_id
from constants import SAMPLE_RATE
from last.speech_features import get_mfcc_features_390
from preprocess import pre_process_inputs, load_wav
from utils import parallel_function, ensures_dir, find_files

logger = logging.getLogger(__name__)


class MiniBatch:
    def __init__(self, libri: pd.DataFrame, batch_size):
        # indices = np.random.choice(len(libri), size=batch_size, replace=False)
        # [anc1, anc2, anc3, pos1, pos2, pos3, neg1, neg2, neg3]
        # [sp1, sp2, sp3, sp1, sp2, sp3, sp4, sp5, sp6]

        unique_speakers = list(libri['speaker_id'].unique())
        num_triplets = batch_size

        anchor_batch = None
        positive_batch = None
        negative_batch = None
        for ii in range(num_triplets):
            two_different_speakers = np.random.choice(unique_speakers, size=2, replace=False)
            anchor_positive_speaker = two_different_speakers[0]
            negative_speaker = two_different_speakers[1]
            anchor_positive_file = libri[libri['speaker_id'] == anchor_positive_speaker].sample(n=2, replace=False)
            anchor_df = pd.DataFrame(anchor_positive_file[0:1])
            anchor_df['training_type'] = 'anchor'
            positive_df = pd.DataFrame(anchor_positive_file[1:2])
            positive_df['training_type'] = 'positive'
            negative_df = libri[libri['speaker_id'] == negative_speaker].sample(n=1)
            negative_df['training_type'] = 'negative'

            if anchor_batch is None:
                anchor_batch = anchor_df.copy()
            else:
                anchor_batch = pd.concat([anchor_batch, anchor_df], axis=0)

            if positive_batch is None:
                positive_batch = positive_df.copy()
            else:
                positive_batch = pd.concat([positive_batch, positive_df], axis=0)

            if negative_batch is None:
                negative_batch = negative_df.copy()
            else:
                negative_batch = pd.concat([negative_batch, negative_df], axis=0)

        self.libri_batch = pd.DataFrame(pd.concat([anchor_batch, positive_batch, negative_batch], axis=0))
        self.audio_loaded = False
        self.num_triplets = num_triplets

    def load_wav(self):
        self.libri_batch = load_wav(self.libri_batch)
        self.audio_loaded = True

    def to_inputs(self):

        if not self.audio_loaded:
            self.load_wav()

        x = self.libri_batch['raw_audio'].values
        new_x = []
        for sig in x:
            new_x.append(pre_process_inputs(sig, target_sample_rate=SAMPLE_RATE))
        x = np.array(new_x)
        y = self.libri_batch['speaker_id'].values
        logging.info('x.shape = {}'.format(x.shape))
        logging.info('y.shape = {}'.format(y.shape))

        # anchor examples [speakers] == positive examples [speakers]
        np.testing.assert_array_equal(y[0:self.num_triplets], y[self.num_triplets:2 * self.num_triplets])

        return x, y


def stochastic_mini_batch(libri, batch_size):
    mini_batch = MiniBatch(libri, batch_size)
    return mini_batch


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


def generate_features(audio_entities, max_count, progress_bar=False):
    features = []
    count_range = range(max_count)
    if progress_bar:
        from tqdm import tqdm
        count_range = tqdm(count_range)
    for _ in count_range:
        audio_entity = np.random.choice(audio_entities)
        voice_only_signal = audio_entity['audio_voice_only']
        # just add a bit of randomness here.
        cut = np.random.uniform(low=0, high=SAMPLE_RATE // 10, size=1)
        signal_to_process = voice_only_signal[int(cut):]
        features_per_conv = get_mfcc_features_390(signal_to_process, SAMPLE_RATE, max_frames=None)
        features.append(features_per_conv)
    return features


def normalize(list_matrices, mean, std):
    return [(m - mean) / std for m in list_matrices]


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

        # if speaker_id not in c.AUDIO.SPEAKERS_TRAINING_SET:
        #     logger.info(f'Discarding speaker for the training dataset (cf. conf.json): {speaker_id}.')
        #     return

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
        feat = generate_features(audio_entities, self.max_count_per_class, progress_bar=False)
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
        cutoff = int(len(audio_entities) * 0.8)
        audio_entities_train = audio_entities[0:cutoff]
        audio_entities_test = audio_entities[cutoff:]

        train = generate_features(audio_entities_train, self.max_count_per_class)
        test = generate_features(audio_entities_test, self.max_count_per_class)
        logger.info(f'Generated {self.max_count_per_class}/{self.max_count_per_class} '
                    f'inputs for train/test for speaker {speaker_id}.')

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
        return inputs


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


def generate_cache_from_training_inputs(cache_dir, audio_reader, parallel):
    inputs_generator = InputsGenerator(
        cache_dir=cache_dir,
        audio_reader=audio_reader,
        max_count_per_class=50,
        speakers_sub_list=None,
        parallel=parallel
    )
    inputs_generator.start_generation()
