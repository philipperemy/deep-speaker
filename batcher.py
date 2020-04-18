import logging
import os
from random import choice

import dill
import numpy as np
from tqdm import tqdm

from audio import pad_mfcc, Audio
from constants import NUM_FRAMES, NUM_FBANKS, TRAIN_TEST_RATIO
from conv_models import DeepSpeakerModel
from utils import ensures_dir, load_pickle, load_npy

logger = logging.getLogger(__name__)


class KerasFormatConverter:

    def __init__(self, working_dir, load_test_only=False):
        self.working_dir = working_dir
        self.output_dir = os.path.join(self.working_dir, 'keras-inputs')
        ensures_dir(self.output_dir)
        self.categorical_speakers = load_pickle(os.path.join(self.output_dir, 'categorical_speakers.pkl'))
        if not load_test_only:
            self.kx_train = load_npy(os.path.join(self.output_dir, 'kx_train.npy'))
            self.ky_train = load_npy(os.path.join(self.output_dir, 'ky_train.npy'))
        self.kx_test = load_npy(os.path.join(self.output_dir, 'kx_test.npy'))
        self.ky_test = load_npy(os.path.join(self.output_dir, 'ky_test.npy'))
        self.audio = Audio(cache_dir=self.working_dir, audio_dir=None)
        if self.categorical_speakers is None:
            self.categorical_speakers = SparseCategoricalSpeakers(self.audio.speaker_ids)

    def persist_to_disk(self):
        with open(os.path.join(self.output_dir, 'categorical_speakers.pkl'), 'wb') as w:
            dill.dump(self.categorical_speakers, w)
        np.save(os.path.join(self.output_dir, 'kx_train.npy'), self.kx_train)
        np.save(os.path.join(self.output_dir, 'kx_test.npy'), self.kx_test)
        np.save(os.path.join(self.output_dir, 'ky_train.npy'), self.ky_train)
        np.save(os.path.join(self.output_dir, 'ky_test.npy'), self.ky_test)

    def generate_per_phase(self, max_length=NUM_FRAMES, num_per_speaker=3000, is_test=False):
        # train OR test.
        num_speakers = len(self.audio.speaker_ids)
        sp_to_utt = {}
        for speaker_id, utterances in self.audio.speakers_to_utterances.items():
            utterances_files = sorted(utterances.values())
            train_test_sep = int(len(utterances_files) * TRAIN_TEST_RATIO)
            sp_to_utt[speaker_id] = utterances_files[train_test_sep:] if is_test else utterances_files[:train_test_sep]

        # 64 fbanks 1 channel(s).
        # float32
        kx = np.zeros((num_speakers * num_per_speaker, max_length, NUM_FBANKS, 1), dtype=np.float32)
        ky = np.zeros((num_speakers * num_per_speaker, 1), dtype=np.float32)

        for i, speaker_id in enumerate(tqdm(self.audio.speaker_ids, desc='Converting to Keras format')):
            utterances_files = sp_to_utt[speaker_id]
            for j, utterance_file in enumerate(np.random.choice(utterances_files, size=num_per_speaker, replace=True)):
                self.load_into_mat(utterance_file, self.categorical_speakers, speaker_id, max_length, kx, ky,
                                   i * num_per_speaker + j)
        return kx, ky

    def generate(self, max_length=NUM_FRAMES, counts_per_speaker=(3000, 500)):
        kx_train, ky_train = self.generate_per_phase(max_length, counts_per_speaker[0], is_test=False)
        kx_test, ky_test = self.generate_per_phase(max_length, counts_per_speaker[1], is_test=True)
        logger.info(f'kx_train.shape = {kx_train.shape}')
        logger.info(f'ky_train.shape = {ky_train.shape}')
        logger.info(f'kx_test.shape = {kx_test.shape}')
        logger.info(f'ky_test.shape = {ky_test.shape}')
        self.kx_train, self.ky_train, self.kx_test, self.ky_test = kx_train, ky_train, kx_test, ky_test

    @staticmethod
    def load_into_mat(utterance_file, categorical_speakers, speaker_id, max_length, kx, ky, i):
        mfcc = np.load(utterance_file)
        y = categorical_speakers.get_index(speaker_id)
        if mfcc.shape[0] >= max_length:
            r = choice(range(0, len(mfcc) - max_length + 1))
            kx[i] = np.expand_dims(mfcc[r:r + max_length], axis=-1)
        else:
            kx[i] = np.expand_dims(pad_mfcc(mfcc, max_length), axis=-1)
        ky[i] = y


class SparseCategoricalSpeakers:

    def __init__(self, speakers_list):
        self.speaker_ids = sorted(speakers_list)
        assert len(set(self.speaker_ids)) == len(self.speaker_ids)  # all unique.
        self.map = dict(zip(self.speaker_ids, range(len(self.speaker_ids))))

    def get_index(self, speaker_id):
        return self.map[speaker_id]


class OneHotSpeakers:

    def __init__(self, speakers_list):
        from tensorflow.keras.utils import to_categorical
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
