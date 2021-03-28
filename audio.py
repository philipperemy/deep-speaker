import logging
import os
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
from python_speech_features import fbank
from tqdm import tqdm

from constants import SAMPLE_RATE, NUM_FBANKS
from utils import find_files, ensures_dir

logger = logging.getLogger(__name__)


def pad_mfcc(mfcc: np.array, max_length: int):
    # pad MFCC with 0.0. if max_length = 160 (default settings), then less than 1.6s of speech will require padding.
    if len(mfcc) < max_length:
        mfcc = np.vstack((mfcc, np.tile(np.zeros(mfcc.shape[1]), (max_length - len(mfcc), 1))))
    return mfcc


def mfcc_fbank(signal: np.array, sample_rate: int):  # 1D signal array.
    assert len(signal.shape) == 1
    # Returns MFCC with shape (num_frames, n_filters, 3).
    filter_banks, energies = fbank(signal, samplerate=sample_rate, nfilt=NUM_FBANKS)
    frames_features = normalize_mfcc_frames(filter_banks)
    # delta_1 = delta(filter_banks, N=1)
    # delta_2 = delta(delta_1, N=1)
    # frames_features = np.transpose(np.stack([filter_banks, delta_1, delta_2]), (1, 2, 0))
    return np.array(frames_features, dtype=np.float32)  # Float32 precision is enough here.


def normalize_mfcc_frames(m: np.array, epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]


def read_mfcc(input_filename, sample_rate):
    audio = Audio.read(input_filename, sample_rate)
    # TODO: could use trim_silence() here or a better VAD.
    energy = np.abs(audio)
    silence_threshold = np.percentile(energy, 95)
    offsets = np.where(energy > silence_threshold)[0]
    audio_voice_only = audio[offsets[0]:offsets[-1]]
    mfcc = mfcc_fbank(audio_voice_only, sample_rate)
    return mfcc


def extract_speaker_and_utterance_ids(libri_filename: str):  # LIBRI.
    # 'audio/dev-other/116/288045/116-288045-0000.flac'
    speaker, _, basename = Path(libri_filename).parts[-3:]
    utterance = os.path.splitext(basename.split('-', 1)[-1])[0]
    assert basename.split('-')[0] == speaker
    return speaker, utterance


class Audio:

    def __init__(self, cache_dir: str, audio_dir: str = None, sample_rate: int = SAMPLE_RATE, ext='flac'):
        self.ext = ext
        self.cache_dir = os.path.join(cache_dir, 'audio-fbanks')
        ensures_dir(self.cache_dir)
        if audio_dir is not None:
            self.build_cache(os.path.expanduser(audio_dir), sample_rate)
        self.speakers_to_utterances = defaultdict(dict)
        for cache_file in find_files(self.cache_dir, ext='npy'):
            # /path/to/speaker_utterance.npy
            speaker_id, utterance_id = Path(cache_file).stem.split('_')
            self.speakers_to_utterances[speaker_id][utterance_id] = cache_file

    @property
    def speaker_ids(self):
        return sorted(self.speakers_to_utterances)

    @staticmethod
    def read(filename, sample_rate=SAMPLE_RATE):
        audio, sr = librosa.load(filename, sr=sample_rate, mono=True, dtype=np.float32)
        assert sr == sample_rate
        return audio

    def build_cache(self, audio_dir, sample_rate):
        logger.info(f'Audio directory : {audio_dir}.')
        logger.info(f'Sample rate     : {sample_rate:,} hz.')
        audio_files = find_files(audio_dir, ext=self.ext)
        audio_files_count = len(audio_files)
        assert audio_files_count != 0, f'Could not find any {self.ext} files in {audio_dir}.'
        logger.info(f'Found {audio_files_count:,} files in {audio_dir}.')
        with tqdm(audio_files) as bar:
            for audio_filename in bar:
                bar.set_description(audio_filename)
                self.cache_audio_file(audio_filename, sample_rate)

    def cache_audio_file(self, input_filename, sample_rate):
        sp, utt = extract_speaker_and_utterance_ids(input_filename)
        cache_filename = os.path.join(self.cache_dir, f'{sp}_{utt}.npy')
        if not os.path.isfile(cache_filename):
            try:
                mfcc = read_mfcc(input_filename, sample_rate)
                np.save(cache_filename, mfcc)
            except librosa.util.exceptions.ParameterError as e:
                logger.error(e)
