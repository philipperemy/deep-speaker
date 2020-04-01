import logging
import os
import pickle
from collections import defaultdict

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import SAMPLE_RATE
from utils import find_files, ensures_dir
from utils import parallel_function

logger = logging.getLogger(__name__)


def read_audio(filename, sample_rate=SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    assert sr == sample_rate
    return audio


def read_librispeech_structure(directory):
    libri = pd.DataFrame()
    libri['filename'] = find_files(directory)
    libri['filename'] = libri['filename'].apply(lambda x: x.replace('\\', '/'))  # normalize windows paths
    libri['chapter_id'] = libri['filename'].apply(lambda x: x.split('/')[-2])
    libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-3])
    libri['dataset_id'] = libri['filename'].apply(lambda x: x.split('/')[-4])
    num_speakers = len(libri['speaker_id'].unique())
    logging.info(f'Found {str(len(libri)).zfill(7)} files with {str(num_speakers).zfill(5)} different speakers.')
    logging.info(libri.head(10))
    return libri


def trim_silence(audio, threshold):
    """Removes silence at the beginning and end of a sample."""
    energy = librosa.feature.rms(audio)
    frames = np.nonzero(np.array(energy > threshold))
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    audio_trim = audio[0:0]
    left_blank = audio[0:0]
    right_blank = audio[0:0]
    if indices.size:
        audio_trim = audio[indices[0]:indices[-1]]
        left_blank = audio[:indices[0]]  # slice before.
        right_blank = audio[indices[-1]:]  # slice after.
    return audio_trim, left_blank, right_blank


def get_mfcc_features_390(sig, rate, max_frames=None):
    window_length_sec = 25.0 / 1000
    window_step_sec = 10.0 / 1000
    window_cnn_fr_size = int(window_length_sec * rate)  # window size in frames
    window_cnn_fr_steps = int(window_step_sec * rate)  # the step size in frames. if step < window, overlap!
    feat_mat = []
    for i in range(int(len(sig) / window_cnn_fr_steps)):
        start = window_cnn_fr_steps * i
        end = start + window_cnn_fr_size
        slice_sig = sig[start:end]
        if len(slice_sig) / rate == window_length_sec:
            feat = mfcc_features(slice_sig, rate).flatten()
            feat_mat.append(feat)
    feat_mat = np.array(feat_mat, dtype=float)

    indices = np.array(range(10))
    new_feat_mat = []
    for frame_id in range(len(feat_mat)):
        if max(indices) >= len(feat_mat):
            break
        new_feat_mat.append(np.transpose(feat_mat[indices]).flatten())  # (39, 10).flatten()
        indices += 3
    new_feat_mat = np.array(new_feat_mat)
    if max_frames is not None:
        new_feat_mat = new_feat_mat[0:max_frames]
    return new_feat_mat


def mfcc_features(sig, rate, nb_features=13):
    from python_speech_features import mfcc, delta
    mfcc_feat = mfcc(sig, rate, numcep=nb_features, nfilt=nb_features)
    delta_feat = delta(mfcc_feat, 2)
    double_delta_feat = delta(delta_feat, 2)
    return np.concatenate((mfcc_feat, delta_feat, double_delta_feat), axis=1)


def extract_speaker_id(filename):
    return filename.split('/')[-2]


def extract_sentence_id(filename):
    return filename.split('/')[-1].split('_')[1].split('.')[0]


class AudioReader:
    SENTENCE_ID = 'sentence_id'
    SPEAKER_ID = 'speaker_id'
    FILENAME = 'filename'

    def __init__(self, input_audio_dir,
                 output_cache_dir,
                 sample_rate,
                 multi_threading=False):
        self.audio_dir = os.path.expanduser(input_audio_dir)
        self.cache_dir = os.path.expanduser(output_cache_dir)
        self.sample_rate = sample_rate
        self.multi_threading = multi_threading
        self.cache_pkl_dir = os.path.join(self.cache_dir, 'audio_cache_pkl')
        self.pkl_filenames = find_files(self.cache_pkl_dir, 'pkl')

        logger.info(f'audio_dir = {self.audio_dir}')
        logger.info(f'cache_dir = {self.cache_dir}')
        logger.info(f'sample_rate = {sample_rate}')

        speakers = set()
        self.speaker_ids_to_filename = {}
        for pkl_filename in self.pkl_filenames:
            speaker_id = os.path.basename(pkl_filename).split('_')[0]
            if speaker_id not in self.speaker_ids_to_filename:
                self.speaker_ids_to_filename[speaker_id] = []
            self.speaker_ids_to_filename[speaker_id].append(pkl_filename)
            speakers.add(speaker_id)
        self.all_speaker_ids = sorted(speakers)

    def load_cache(self, speakers_sub_list=None):
        cache = {}
        metadata = defaultdict(dict)

        if speakers_sub_list is None:
            filenames = self.pkl_filenames
        else:
            filenames = []
            for speaker_id in speakers_sub_list:
                filenames.extend(self.speaker_ids_to_filename[speaker_id])

        for pkl_file in filenames:
            with open(pkl_file, 'rb') as f:
                obj = pickle.load(f)
                if self.FILENAME in obj:
                    cache[obj[self.FILENAME]] = obj

        for filename in sorted(cache):
            speaker_id = extract_speaker_id(filename)
            sentence_id = extract_sentence_id(filename)
            if sentence_id not in metadata[speaker_id]:
                metadata[speaker_id][sentence_id] = []
            metadata[speaker_id][sentence_id] = {
                self.SPEAKER_ID: speaker_id,
                self.SENTENCE_ID: sentence_id,
                self.FILENAME: filename
            }

        # metadata # small cache <speaker_id -> sentence_id, filename> - auto generated from self.cache.
        # cache # big cache <filename, data:audio librosa, blanks.>
        return cache, metadata

    def build_cache(self):
        ensures_dir(self.cache_pkl_dir)
        logger.info(f'Nothing found at {self.cache_pkl_dir}. Generating all the cache now.')
        logger.info(f'Looking for the audio dataset in {self.audio_dir}.')
        audio_files = find_files(self.audio_dir, extension='wav')
        audio_files_count = len(audio_files)
        assert audio_files_count != 0, f'Could not find any WAV files in {self.audio_dir}.'
        logger.info(f'Found {audio_files_count:,} files in total in {self.audio_dir}.')
        if self.multi_threading:
            num_threads = os.cpu_count()
            parallel_function(self.cache_audio_file, audio_files, num_threads)
        else:
            with tqdm(audio_files) as bar:
                for filename in bar:
                    bar.set_description(filename)
                    self.cache_audio_file(filename)

    def cache_audio_file(self, input_filename):
        try:
            cache_filename = os.path.splitext(os.path.basename(input_filename))[0] + '_cache'
            pkl_filename = os.path.join(self.cache_pkl_dir, cache_filename) + '.pkl'

            if os.path.isfile(pkl_filename):
                logger.info('[FILE ALREADY EXISTS] {}'.format(pkl_filename))
                return

            audio = read_audio(input_filename, self.sample_rate)
            energy = np.abs(audio)
            silence_threshold = np.percentile(energy, 95)
            offsets = np.where(energy > silence_threshold)[0]
            left_blank_duration_ms = (1000.0 * offsets[0]) // self.sample_rate  # frame_id to duration (ms)
            right_blank_duration_ms = (1000.0 * (len(audio) - offsets[-1])) // self.sample_rate
            # _, left_blank, right_blank = trim_silence(audio[:, 0], silence_threshold)
            # logger.info('_' * 100)
            # logger.info('left_blank_duration_ms = {}, right_blank_duration_ms = {}, '
            #             'audio_length = {} frames, silence_threshold = {}'.format(left_blank_duration_ms,
            #                                                                       right_blank_duration_ms,
            #                                                                       len(audio),
            #                                                                       silence_threshold))
            obj = {
                'audio': audio,
                'audio_voice_only': audio[offsets[0]:offsets[-1]],
                'left_blank_duration_ms': left_blank_duration_ms,
                'right_blank_duration_ms': right_blank_duration_ms,
                self.FILENAME: input_filename
            }

            with open(pkl_filename, 'wb') as f:
                pickle.dump(obj, f)
                logger.info('[DUMP AUDIO] {}'.format(pkl_filename))
        except librosa.util.exceptions.ParameterError as e:
            logger.error(e)
            logger.error('[DUMP AUDIO ERROR SKIPPING FILENAME] {}'.format(input_filename))
