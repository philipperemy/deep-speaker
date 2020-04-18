import logging
import os
from collections import defaultdict
from random import choice

import dill
import numpy as np
from keras.utils import to_categorical
from python_speech_features import fbank
from tqdm import tqdm

from audio import extract_speaker_id
from constants import SAMPLE_RATE, TRAIN_TEST_RATIO, NUM_FRAMES, NUM_FBANKS
from conv_models import DeepSpeakerModel
from utils import parallel_function, ensures_dir, find_files

logger = logging.getLogger(__name__)


# TODO: try the random batcher. without any logic.
# TODO: try to mine the hard triplets.


def normalize_frames(m, epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]


def mfcc_fbank(signal: np.array, sample_rate: int):  # 1D signal array.
    # Returns MFCC with shape (num_frames, n_filters, 3).
    filter_banks, energies = fbank(signal, samplerate=sample_rate, nfilt=NUM_FBANKS)
    frames_features = normalize_frames(filter_banks)
    # delta_1 = delta(filter_banks, N=1)
    # delta_2 = delta(delta_1, N=1)
    # frames_features = np.transpose(np.stack([filter_banks, delta_1, delta_2]), (1, 2, 0))
    return np.array(frames_features, dtype=np.float32)  # Float32 precision is enough here.


def pad_mfcc(mfcc, max_length):  # num_frames, nfilt=64.
    if len(mfcc) < max_length:
        mfcc = np.vstack((mfcc, np.tile(np.zeros(mfcc.shape[1]), (max_length - len(mfcc), 1))))
    return mfcc


def get_mfcc_from_list(audio_entities):
    mfcc_samples = []
    # with default params, winlen=0.025, winstep=0.01, 1 seconds ~ 100 frames.
    for audio_entity in audio_entities:
        try:
            mfcc_samples.append(mfcc_fbank(audio_entity['audio_voice_only'], SAMPLE_RATE))
        except IndexError:  # happens if signal is too small.
            logger.warning(f'Could not compute the fbank for this {audio_entity["filename"]} (after VAD processing).')
    return mfcc_samples


def normalize(list_matrices, mean, std):
    return [(m - mean) / std for m in list_matrices]


class KerasConverter:

    def __init__(self, working_dir, load_test_only=False):
        self.working_dir = working_dir
        self.output_dir = os.path.join(self.working_dir, 'keras-inputs')
        ensures_dir(self.output_dir)
        self.categorical_speakers = None
        self.kx_train = None
        self.kx_test = None
        self.ky_train = None
        self.ky_test = None
        self.load_from_disk(load_test_only)

    def load_from_disk(self, load_test_only):

        def load_pickle(file):
            if not os.path.exists(file):
                return None
            logger.info(f'Loading PKL file: {file}.')
            with open(file, 'rb') as r:
                return dill.load(r)

        def load_npy(file):
            if not os.path.exists(file):
                return None
            logger.info(f'Loading NPY file: {file}.')
            return np.load(file)

        self.categorical_speakers = load_pickle(os.path.join(self.output_dir, 'categorical_speakers.pkl'))
        if not load_test_only:
            self.kx_train = load_npy(os.path.join(self.output_dir, 'kx_train.npy'))
            self.ky_train = load_npy(os.path.join(self.output_dir, 'ky_train.npy'))
        self.kx_test = load_npy(os.path.join(self.output_dir, 'kx_test.npy'))
        self.ky_test = load_npy(os.path.join(self.output_dir, 'ky_test.npy'))

    def persist_to_disk(self):
        with open(os.path.join(self.output_dir, 'categorical_speakers.pkl'), 'wb') as w:
            dill.dump(self.categorical_speakers, w)
        np.save(os.path.join(self.output_dir, 'kx_train.npy'), self.kx_train)
        np.save(os.path.join(self.output_dir, 'kx_test.npy'), self.kx_test)
        np.save(os.path.join(self.output_dir, 'ky_train.npy'), self.ky_train)
        np.save(os.path.join(self.output_dir, 'ky_test.npy'), self.ky_test)

    def generate(self, max_length=NUM_FRAMES, counts_per_speaker=(3000, 500)):
        fbank_files = os.path.join(self.working_dir, 'fbank-inputs')
        speakers_list = [os.path.splitext(os.path.basename(a))[0] for a in find_files(fbank_files, ext='pkl')]
        categorical_speakers = OneHotSpeakers(speakers_list)
        kx_train, ky_train, kx_test, ky_test = None, None, None, None
        c_train, c_test = 0, 0
        invalid_c_train, invalid_c_test = 0, 0
        for speaker_id in tqdm(categorical_speakers.get_speaker_ids(), desc='Converting to Keras format'):
            with open(os.path.join(self.working_dir, 'fbank-inputs', speaker_id + '.pkl'), 'rb') as r:
                d = dill.load(r)
            y = categorical_speakers.get_one_hot(d['speaker_id'])

            if kx_train is None:
                num_samples_train = len(speakers_list) * counts_per_speaker[0]
                num_samples_test = len(speakers_list) * counts_per_speaker[1]

                # 64 fbanks 3 channels.
                # float32
                kx_train = np.zeros((num_samples_train, max_length, NUM_FBANKS, 1), dtype=np.float32)
                ky_train = np.zeros((num_samples_train, len(speakers_list)), dtype=np.float32)

                kx_test = np.zeros((num_samples_test, max_length, NUM_FBANKS, 1), dtype=np.float32)
                ky_test = np.zeros((num_samples_test, len(speakers_list)), dtype=np.float32)

            # TRAIN
            c = 0
            cond = True
            while cond:
                for x_train_elt in d['train']:
                    if c >= counts_per_speaker[0]:
                        cond = False
                        break
                    if len(x_train_elt) >= max_length:
                        # TODO: we should slice and input that to the model.
                        st = choice(range(0, len(x_train_elt) - max_length + 1))
                        kx_train[c_train] = np.expand_dims(x_train_elt[st:st + max_length], axis=-1)

                    else:
                        kx_train[c_train] = np.expand_dims(pad_mfcc(x_train_elt, max_length), axis=-1)
                        # simple for now.
                        invalid_c_train += 1
                    ky_train[c_train] = y
                    c_train += 1
                    c += 1

            # TEST
            c = 0
            cond = True
            while cond:
                for x_test_elt in d['test']:
                    if c >= counts_per_speaker[1]:
                        cond = False
                        break
                    if len(x_test_elt) >= max_length:
                        # TODO: we should slice and input that to the model.
                        st = choice(range(0, len(x_test_elt) - max_length + 1))
                        kx_test[c_test] = np.expand_dims(x_test_elt[st:st + max_length], axis=-1)
                    else:
                        kx_test[c_test] = np.expand_dims(pad_mfcc(x_test_elt, max_length), axis=-1)
                        invalid_c_test += 1
                    ky_test[c_test] = y
                    c_test += 1
                    c += 1

        logger.info(f'kx_train.shape = {kx_train.shape}')
        logger.info(f'kx_test.shape = {kx_test.shape}')
        logger.info(f'ky_train.shape = {ky_train.shape}')
        logger.info(f'ky_test.shape = {ky_test.shape}')

        pad_ratio_train = (c_train - invalid_c_train) / c_train
        pad_ratio_test = (c_test - invalid_c_test) / c_test
        logger.info(f'Pad ratio train (1 is perfect): {pad_ratio_train:.4f}.')
        logger.info(f'Pad ratio test (1 is perfect): {pad_ratio_test:.4f}.')

        # if pad_ratio_train < 0.9:
        #     logger.warning(f'Many audio are padded for train: Set a lower value for NUM_FRAMES.')
        #     exit(1)
        #
        # if pad_ratio_test < 0.9:
        #     logger.warning(f'Low mix ratio for test: Set a lower value for NUM_FRAMES.')
        #     exit(1)

        assert c_train == len(ky_train)
        assert c_test == len(ky_test)
        self.categorical_speakers = categorical_speakers
        self.kx_train, self.ky_train, self.kx_test, self.ky_test = kx_train, ky_train, kx_test, ky_test


class FBankProcessor:

    def __init__(self, working_dir, audio_reader, speakers_sub_list=None, parallel=False):
        self.working_dir = os.path.expanduser(working_dir)
        self.audio_reader = audio_reader
        self.parallel = parallel
        self.model_inputs_dir = os.path.join(self.working_dir, 'fbank-inputs')
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

        train = get_mfcc_from_list(audio_entities_train)
        test = get_mfcc_from_list(audio_entities_test)

        inputs = {
            'train': train,
            'test': test,
            'speaker_id': speaker_id,
        }

        with open(output_filename, 'wb') as w:
            dill.dump(obj=inputs, file=w)
        logger.info(f'FBanks generated for speaker: ({speaker_id}, {output_filename}).')

    def generate_inputs_for_inference(self, speaker_id):
        speaker_cache, metadata = self.audio_reader.load_cache([speaker_id])
        audio_entities = list(speaker_cache.values())
        logger.info(f'Generating the inputs necessary for the inference (speaker is {speaker_id})...')
        logger.info('This might take a couple of minutes to complete.')
        feat = get_mfcc_from_list(audio_entities)
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


class TripletBatcher:

    def __init__(self, kx_train, ky_train, kx_test, ky_test):
        self.kx_train = kx_train
        self.ky_train = ky_train
        self.kx_test = kx_test
        self.ky_test = ky_test
        speakers_list = sorted(set(ky_train.argmax(axis=1)))
        num_different_speakers = len(speakers_list)
        assert speakers_list == sorted(set(ky_test.argmax(axis=1)))  # train speakers = test speakers.
        assert speakers_list == list(range(num_different_speakers))
        self.train_indices_per_speaker = {}
        self.test_indices_per_speaker = {}

        for speaker_id in speakers_list:
            self.train_indices_per_speaker[speaker_id] = list(np.where(ky_train.argmax(axis=1) == speaker_id)[0])
            self.test_indices_per_speaker[speaker_id] = list(np.where(ky_test.argmax(axis=1) == speaker_id)[0])

        # check.
        # print(sorted(sum([v for v in self.train_indices_per_speaker.values()], [])))
        # print(range(len(ky_train)))
        assert sorted(sum([v for v in self.train_indices_per_speaker.values()], [])) == sorted(range(len(ky_train)))
        assert sorted(sum([v for v in self.test_indices_per_speaker.values()], [])) == sorted(range(len(ky_test)))
        self.speakers_list = speakers_list

    def select_speaker_data(self, speaker, n, is_test):
        x = self.kx_test if is_test else self.kx_train
        indices_per_speaker = self.test_indices_per_speaker if is_test else self.train_indices_per_speaker
        indices = np.random.choice(indices_per_speaker[speaker], size=n)
        return x[indices]

    def get_batch(self, batch_size, is_test=False):
        # y = self.ky_test if is_test else self.ky_train

        two_different_speakers = np.random.choice(self.speakers_list, size=2, replace=False)
        anchor_positive_speaker = two_different_speakers[0]
        negative_speaker = two_different_speakers[1]
        assert negative_speaker != anchor_positive_speaker

        batch_x = np.vstack([
            self.select_speaker_data(anchor_positive_speaker, batch_size // 3, is_test),
            self.select_speaker_data(anchor_positive_speaker, batch_size // 3, is_test),
            self.select_speaker_data(negative_speaker, batch_size // 3, is_test)
        ])

        batch_y = np.zeros(shape=(len(batch_x), len(self.speakers_list)))
        return batch_x, batch_y


class TripletBatcherMiner(TripletBatcher):

    def __init__(self, kx_train, ky_train, kx_test, ky_test, model: DeepSpeakerModel):
        super().__init__(kx_train, ky_train, kx_test, ky_test)
        self.model = model
        self.num_evaluations_to_find_best_batch = 10

    def get_batch(self, batch_size, is_test=False):
        if is_test:
            return super().get_batch(batch_size, is_test)
        max_loss = 0
        max_batch = None, None
        for i in range(self.num_evaluations_to_find_best_batch):
            bx, by = super().get_batch(batch_size, is_test=False)  # only train here.
            loss = self.model.m.evaluate(bx, by, batch_size=batch_size, verbose=0)
            if loss > max_loss:
                max_loss = loss
                max_batch = bx, by
        return max_batch


class TripletBatcherSelectHardNegatives(TripletBatcher):

    def __init__(self, kx_train, ky_train, kx_test, ky_test, model: DeepSpeakerModel):
        super().__init__(kx_train, ky_train, kx_test, ky_test)
        self.model = model

    def get_batch(self, batch_size, is_test=False, predict=None):
        if predict is None:
            predict = self.model.m.predict
        from test import batch_cosine_similarity
        num_triplets = batch_size // 3
        inputs = []
        k = 2  # do not change this.
        for speaker in self.speakers_list:
            inputs.append(self.select_speaker_data(speaker, n=k, is_test=is_test))
        inputs = np.array(inputs)  # num_speakers * [k, num_frames, num_fbanks, 1].
        embeddings = predict(np.vstack(inputs))
        assert embeddings.shape[-1] == 512
        # (speaker, utterance, 512)
        embeddings = np.reshape(embeddings, (len(self.speakers_list), k, 512))
        cs = batch_cosine_similarity(embeddings[:, 0], embeddings[:, 1])
        arg_sort = np.argsort(cs)
        assert len(arg_sort) > num_triplets
        anchor_speakers = arg_sort[0:num_triplets]

        anchor_embeddings = embeddings[anchor_speakers, 0]
        negative_speakers = sorted(set(self.speakers_list) - set(anchor_speakers))
        negative_embeddings = embeddings[negative_speakers, 0]

        selected_negative_speakers = []
        for anchor_embedding in anchor_embeddings:
            cs_negative = [batch_cosine_similarity([anchor_embedding], neg) for neg in negative_embeddings]
            selected_negative_speakers.append(negative_speakers[int(np.argmax(cs_negative))])

        # anchor with frame 0.
        # positive with frame 1.
        # negative with frame 0.
        assert len(set(selected_negative_speakers).intersection(anchor_speakers)) == 0
        negative = inputs[selected_negative_speakers, 0]
        positive = inputs[anchor_speakers, 1]
        anchor = inputs[anchor_speakers, 0]
        batch_x = np.vstack([anchor, positive, negative])
        batch_y = np.zeros(shape=(len(batch_x), len(self.speakers_list)))
        return batch_x, batch_y


class TripletEvaluator:

    def __init__(self, kx_test, ky_test):
        self.kx_test = kx_test
        self.ky_test = ky_test
        speakers_list = sorted(set(ky_test.argmax(axis=1)))
        num_different_speakers = len(speakers_list)
        assert speakers_list == list(range(num_different_speakers))
        self.test_indices_per_speaker = {}
        for speaker_id in speakers_list:
            self.test_indices_per_speaker[speaker_id] = list(np.where(ky_test.argmax(axis=1) == speaker_id)[0])
        assert sorted(sum([v for v in self.test_indices_per_speaker.values()], [])) == sorted(range(len(ky_test)))
        self.speakers_list = speakers_list

    def _select_speaker_data(self, speaker):
        indices = np.random.choice(self.test_indices_per_speaker[speaker], size=1)
        return self.kx_test[indices]

    def get_speaker_verification_data(self, positive_speaker, num_different_speakers):
        all_negative_speakers = list(set(self.speakers_list) - {positive_speaker})
        assert len(self.speakers_list) - 1 == len(all_negative_speakers)
        negative_speakers = np.random.choice(all_negative_speakers, size=num_different_speakers, replace=False)
        assert positive_speaker not in negative_speakers
        anchor = self._select_speaker_data(positive_speaker)
        positive = self._select_speaker_data(positive_speaker)
        data = [anchor, positive]
        data.extend([self._select_speaker_data(n) for n in negative_speakers])
        return np.vstack(data)
