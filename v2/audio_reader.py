import logging
import os
import pickle
import re
from glob import glob

import librosa
import numpy as np
from tqdm import tqdm

from utils import parallel_function

logger = logging.getLogger(__name__)

SENTENCE_ID = 'sentence_id'
SPEAKER_ID = 'speaker_id'
FILENAME = 'filename'


class PreMergeProcessor:
    @staticmethod
    def ramp(sound_1, sound_2):
        mask_1 = np.linspace(1, 0, len(sound_1))
        mask_2 = np.linspace(0, 1, len(sound_2))
        return np.expand_dims(np.multiply(mask_1, sound_1.flatten()), axis=1), \
               np.expand_dims(np.multiply(mask_2, sound_2.flatten()), axis=1)


class SoundMerger:
    @staticmethod
    def default_merger(sound_1, sound_2):
        return 0.5 * sound_1 + 0.5 * sound_2

    @staticmethod
    def clip_merger(sound_1, sound_2):
        return np.clip(sound_1 + sound_2, -1, 1)

    @staticmethod
    def add_merger(sound_1, sound_2):
        return sound_1 + sound_2


def find_files(directory, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern."""
    return sorted(glob(directory + pattern, recursive=True))


def read_audio_from_filename(filename, sample_rate):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)
    return audio, filename


def load_generic_audio(files, sample_rate):
    for filename in files:
        audio, filename = read_audio_from_filename(filename, sample_rate)
        yield audio, filename


def load_vctk_audio(directory, sample_rate):
    """Generator that yields audio waveforms from the VCTK dataset, and
    additionally the ID of the corresponding speaker."""
    files = find_files(directory)
    speaker_re = re.compile(r'p([0-9]+)_([0-9]+)\.wav')
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        matches = speaker_re.findall(filename)[0]
        speaker_id, recording_id = [int(id_) for id_ in matches]
        yield audio, speaker_id


def overlap(sound1, sound2, overlap_length,
            merge=SoundMerger.add_merger,
            pre_merge_processor=PreMergeProcessor.ramp):
    tmp = np.array(sound1)
    l1 = len(sound1)
    l2 = len(sound2)
    # print('-> l1 = {}, l2 = {}'.format(l1, l2))
    if overlap_length >= l1 and overlap_length >= l2:
        overlap_length = min(l1, l2)
    elif overlap_length >= l2:
        overlap_length = l2
    elif overlap_length >= l1:
        overlap_length = l1
    overlap_1, overlap_2 = pre_merge_processor(tmp[-overlap_length:], sound2[:overlap_length])
    overlap_part = merge(overlap_1, overlap_2)
    assert len(overlap_part) == overlap_length
    assert np.max(overlap_part) <= 1.0
    assert np.min(overlap_part) >= -1.0
    tmp[-overlap_length:] = overlap_part
    out = np.concatenate((tmp, sound2[overlap_length:]))
    assert len(out) == l1 + l2 - overlap_length
    out = np.clip(out, -1, 1)
    mid_overlap_point_offset = int(l1)
    # mid_overlap_point_offset = int(l1 - overlap_length * 0.5)
    return out, mid_overlap_point_offset


def trim_silence(audio, threshold):
    """Removes silence at the beginning and end of a sample."""
    energy = librosa.feature.rmse(audio)
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


def extract_speaker_id(filename):
    return filename.split('/')[-2]


def extract_sentence_id(filename):
    return filename.split('/')[-1].split('_')[1].split('.')[0]


class AudioReader:
    def __init__(self, input_audio_dir,
                 output_cache_dir,
                 sample_rate,
                 multi_threading=False):
        self.audio_dir = os.path.expanduser(input_audio_dir)
        self.cache_dir = os.path.expanduser(output_cache_dir)
        self.sample_rate = sample_rate
        self.multi_threading = multi_threading
        self.cache_pkl_dir = os.path.join(self.cache_dir, 'audio_cache_pkl')
        self.pkl_filenames = find_files(self.cache_pkl_dir, pattern='/**/*.pkl')

        logger.info('audio_dir = {}'.format(self.audio_dir))
        logger.info('cache_dir = {}'.format(self.cache_dir))
        logger.info('sample_rate = {}'.format(sample_rate))

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
        metadata = {}

        if speakers_sub_list is None:
            filenames = self.pkl_filenames
        else:
            filenames = []
            for speaker_id in speakers_sub_list:
                filenames.extend(self.speaker_ids_to_filename[speaker_id])

        for pkl_file in filenames:
            with open(pkl_file, 'rb') as f:
                obj = pickle.load(f)
                if FILENAME in obj:
                    cache[obj[FILENAME]] = obj

        for filename in sorted(cache):
            speaker_id = extract_speaker_id(filename)
            if speaker_id not in metadata:
                metadata[speaker_id] = {}
            sentence_id = extract_sentence_id(filename)
            if sentence_id not in metadata[speaker_id]:
                metadata[speaker_id][sentence_id] = []
            metadata[speaker_id][sentence_id] = {SPEAKER_ID: speaker_id,
                                                 SENTENCE_ID: sentence_id,
                                                 FILENAME: filename}

        # metadata # small cache <speaker_id -> sentence_id, filename> - auto generated from self.cache.
        # cache # big cache <filename, data:audio librosa, blanks.>
        return cache, metadata

    def build_cache(self):
        if not os.path.exists(self.cache_pkl_dir):
            os.makedirs(self.cache_pkl_dir)
        logger.info('Nothing found at {}. Generating all the cache now.'.format(self.cache_pkl_dir))
        logger.info('Looking for the audio dataset in {}.'.format(self.audio_dir))
        audio_files = find_files(self.audio_dir)
        audio_files_count = len(audio_files)
        assert audio_files_count != 0, 'Generate your cache please.'
        logger.info('Found {} files in total in {}.'.format(audio_files_count, self.audio_dir))
        assert len(audio_files) != 0

        if self.multi_threading:
            num_threads = os.cpu_count()
            parallel_function(self.dump_audio_to_pkl_cache, audio_files, num_threads)
        else:
            bar = tqdm(audio_files)
            for filename in bar:
                bar.set_description(filename)
                self.dump_audio_to_pkl_cache(filename)
            bar.close()

    def dump_audio_to_pkl_cache(self, input_filename):
        try:
            cache_filename = input_filename.split('/')[-1].split('.')[0] + '_cache'
            pkl_filename = os.path.join(self.cache_pkl_dir, cache_filename) + '.pkl'

            if os.path.isfile(pkl_filename):
                logger.info('[FILE ALREADY EXISTS] {}'.format(pkl_filename))
                return

            audio, _ = read_audio_from_filename(input_filename, self.sample_rate)
            energy = np.abs(audio[:, 0])
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
            obj = {'audio': audio,
                   'audio_voice_only': audio[offsets[0]:offsets[-1]],
                   'left_blank_duration_ms': left_blank_duration_ms,
                   'right_blank_duration_ms': right_blank_duration_ms,
                   FILENAME: input_filename}

            with open(pkl_filename, 'wb') as f:
                pickle.dump(obj, f)
                logger.info('[DUMP AUDIO] {}'.format(pkl_filename))
        except librosa.util.exceptions.ParameterError as e:
            logger.error(e)
            logger.error('[DUMP AUDIO ERROR SKIPPING FILENAME] {}'.format(input_filename))
