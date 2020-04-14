import logging
import os
import pickle
from collections import defaultdict

import librosa
import numpy as np
from tqdm import tqdm

from constants import SAMPLE_RATE
from utils import find_files, ensures_dir
from utils import parallel_function

logger = logging.getLogger(__name__)


def extract_speaker_id(filename):
    return filename.split('/')[-2]


def extract_sentence_id(filename):
    return filename.split('/')[-1].split('_')[1].split('.')[0]


class Audio:
    SENTENCE_ID = 'sentence_id'
    SPEAKER_ID = 'speaker_id'
    FILENAME = 'filename'

    def __init__(self, input_audio_dir,
                 output_working_dir,
                 sample_rate,
                 multi_threading=False):
        self.audio_dir = os.path.expanduser(input_audio_dir)
        self.working_dir = os.path.expanduser(output_working_dir)
        self.sample_rate = sample_rate
        self.parallel = multi_threading
        self.cache_pkl_dir = os.path.join(self.working_dir, 'audio_cache')
        self.pkl_filenames = find_files(self.cache_pkl_dir, ext='pkl')
        self.speaker_ids_to_filename = defaultdict(list)

        logger.info(f'audio_dir: {self.audio_dir}.')
        logger.info(f'working_dir: {self.working_dir}.')
        logger.info(f'sample_rate: {self.sample_rate} hz.')

        unique_speakers = set()
        for pkl_filename in self.pkl_filenames:
            speaker_id = os.path.basename(pkl_filename).split('_')[0]
            self.speaker_ids_to_filename[speaker_id].append(pkl_filename)
            unique_speakers.add(speaker_id)
        self.all_speaker_ids = sorted(unique_speakers)

    @staticmethod
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

    @staticmethod
    def read(filename, sample_rate=SAMPLE_RATE):
        audio, sr = librosa.load(filename, sr=sample_rate, mono=True, dtype=np.float32)
        assert sr == sample_rate
        return audio

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
        audio_files = find_files(self.audio_dir, ext='wav')
        audio_files_count = len(audio_files)
        assert audio_files_count != 0, f'Could not find any WAV files in {self.audio_dir}.'
        logger.info(f'Found {audio_files_count:,} files in total in {self.audio_dir}.')
        if self.parallel:
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

            audio = Audio.read(input_filename, self.sample_rate)
            # TODO: could use trim_silence() here or a better VAD.
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
