from time import time

import dill
import librosa
import numpy as np
import os
import re
from glob import glob
from random import shuffle, randint, choice
from tqdm import tqdm

from helpers.logger import Logger

logger = Logger.instance()

TMP_DIR = '/tmp/speaker-change-detection/'
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

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
    return sorted(glob(os.path.join(directory, pattern), recursive=True))


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


class AudioReader(object):
    def __init__(self,
                 audio_dir,
                 sample_rate,
                 speakers_sub_list=None):
        self.audio_dir = os.path.expanduser(audio_dir)  # for the ~/
        self.sample_rate = sample_rate
        self.metadata = dict()  # small cache <SPEAKER_ID -> SENTENCE_ID, filename>
        self.cache = dict()  # big cache <filename, data:audio librosa, blanks.>

        logger.debug('Initializing AudioReader()')
        logger.debug('audio_dir = {}'.format(self.audio_dir))
        logger.debug('sample_rate = {}'.format(sample_rate))
        logger.debug('speakers_sub_list = {}'.format(speakers_sub_list))

        st = time()
        if len(find_files(TMP_DIR, pattern='*.pkl')) == 0:  # generate all the pickle files.
            logger.debug('Nothing found at {}. Generating all the caches now.'.format(TMP_DIR))
            files = find_files(self.audio_dir)
            assert len(files) != 0, 'Generate your cache please.'
            logger.debug('Found {} files in total in {}.'.format(len(files), self.audio_dir))
            if speakers_sub_list is not None:
                files = list(
                    filter(lambda x: any(word in extract_speaker_id(x) for word in speakers_sub_list), files))
                logger.debug('{} files correspond to the speaker list {}.'.format(len(files), speakers_sub_list))
            assert len(files) != 0

            bar = tqdm(files)
            for filename in bar:
                bar.set_description(filename)
                try:
                    speaker_id = extract_speaker_id(filename)
                    audio, _ = read_audio_from_filename(filename, self.sample_rate)
                    energy = np.abs(audio[:, 0])
                    silence_threshold = np.percentile(energy, 95)
                    offsets = np.where(energy > silence_threshold)[0]
                    left_blank_duration_ms = (1000.0 * offsets[0]) // self.sample_rate  # frame_id to duration (ms)
                    right_blank_duration_ms = (1000.0 * (len(audio) - offsets[-1])) // self.sample_rate
                    # _, left_blank, right_blank = trim_silence(audio[:, 0], silence_threshold)
                    logger.info('_' * 100)
                    logger.info('left_blank_duration_ms = {}, right_blank_duration_ms = {}, '
                                'audio_length = {} frames, silence_threshold = {}'.format(left_blank_duration_ms,
                                                                                          right_blank_duration_ms,
                                                                                          len(audio),
                                                                                          silence_threshold))
                    obj = {'audio': audio,
                           'audio_voice_only': audio[offsets[0]:offsets[-1]],
                           'left_blank_duration_ms': left_blank_duration_ms,
                           'right_blank_duration_ms': right_blank_duration_ms,
                           FILENAME: filename}
                    cache_filename = filename.split('/')[-1].split('.')[0] + '_cache'
                    tmp_filename = os.path.join(TMP_DIR, cache_filename) + '.pkl'
                    with open(tmp_filename, 'wb') as f:
                        dill.dump(obj, f)
                        logger.debug('[DUMP AUDIO] {}'.format(tmp_filename))
                    # commit to metadata dictionary when you're sure no errors occurred during processing.
                    if speaker_id not in self.metadata:
                        self.metadata[speaker_id] = {}
                    sentence_id = extract_sentence_id(filename)
                    if sentence_id not in self.metadata[speaker_id]:
                        self.metadata[speaker_id][sentence_id] = []
                    self.metadata[speaker_id][sentence_id] = {SPEAKER_ID: speaker_id,
                                                              SENTENCE_ID: sentence_id,
                                                              FILENAME: filename}
                except librosa.util.exceptions.ParameterError as e:
                    logger.error(e)
                    logger.error('[DUMP AUDIO ERROR SKIPPING FILENAME] {}'.format(filename))
            dill.dump(self.metadata, open(os.path.join(TMP_DIR, 'metadata.pkl'), 'wb'))

        logger.debug(
            'Using the generated files at {}. Using them to load the cache. Be sure to have enough memory.'.format(
                TMP_DIR))
        self.metadata = dill.load(open(os.path.join(TMP_DIR, 'metadata.pkl'), 'rb'))

        pickle_files = find_files(TMP_DIR, pattern='*.pkl')
        for pkl_file in tqdm(pickle_files, desc='reading cache'):
            if 'metadata' not in pkl_file:
                with open(pkl_file, 'rb') as f:
                    obj = dill.load(f)
                    self.cache[obj[FILENAME]] = obj
        logger.debug('Cache took {0:.2f} seconds to load. {1:} keys.'.format(time() - st, len(self.cache)))

    def get_speaker_list(self):
        return sorted(list(self.metadata.keys()))

    def sample_speakers(self, speaker_list, num_speakers):
        if speaker_list is None:
            speaker_list = self.get_speaker_list()
        all_speakers = list(speaker_list)
        shuffle(all_speakers)
        speaker_list = all_speakers[0:num_speakers]
        return speaker_list

    def define_random_mix(self, num_sentences=3, num_speakers=3, speaker_ids_to_choose_from=None):
        speaker_ids_to_choose_from = self.sample_speakers(speaker_ids_to_choose_from, num_speakers)
        targets = []
        for i in range(num_sentences):
            speaker_id = choice(speaker_ids_to_choose_from)
            sentence_id = choice(list(self.metadata[speaker_id].keys()))
            targets.append({SPEAKER_ID: speaker_id,
                            SENTENCE_ID: sentence_id,
                            FILENAME: self.metadata[speaker_id][sentence_id]['filename']})
        return targets

    def define_mix(self, mix_sequence):
        targets = []
        for speaker_sentence_id in mix_sequence:
            speaker_id, sentence_id = speaker_sentence_id

            if isinstance(speaker_id, int):
                speaker_id = 'p' + str(speaker_id)
            if isinstance(sentence_id, int):
                sentence_id = str(sentence_id).zfill(3)

            targets.append({SPEAKER_ID: speaker_id,
                            SENTENCE_ID: sentence_id,
                            FILENAME: self.metadata[speaker_id][sentence_id]['filename']})
        return targets

    def generate_mix_with_voice_only(self, targets):
        if len(targets) == 0:
            return [], None
        audio_dict = {}
        for i, target in enumerate(targets):
            audio_dict[i] = self.cache[target[FILENAME]]
        output = audio_dict[0]['audio_voice_only']
        targets[0]['offset'] = 0  # in ms
        for i in range(1, len(targets)):
            offset_pos = -1
            beg = targets[i - 1]['offset']
            while offset_pos <= beg:
                offset_pos = len(output)
                output = np.concatenate((output, audio_dict[i]['audio_voice_only']))
                targets[i]['offset'] = offset_pos
        return targets, output

    def generate_mix(self, targets):
        audio_dict = {}
        for i, target in enumerate(targets):
            audio_dict[i] = self.cache[target[FILENAME]]

        output = audio_dict[0]['audio']
        targets[0]['offset'] = 0  # in ms
        for i in range(1, len(targets)):
            # example: sound 1 - 600 ms at the end.
            dur_right_blank_1 = audio_dict[i - 1]['right_blank_duration_ms']
            # example: sound 2 - 25 ms at the beginning (for silence).
            dur_left_blank_2 = audio_dict[i]['left_blank_duration_ms']
            blank_duration_ms = int(dur_right_blank_1 + dur_left_blank_2)
            # example: sample from U[1, 600+25ms] 1 and not 0 because otherwise we have a shape error.
            # ValueError: operands could not be broadcast together with shapes (723639,1) (0,1)

            offset_pos = -1
            beg = targets[i - 1]['offset']
            while offset_pos <= beg:
                # print('-> append : {}'.format(audio_dict[i][FILENAME]))
                if blank_duration_ms <= 1:
                    rand_val = 1
                else:
                    rand_val = int(randint(1, blank_duration_ms) * 0.8) + 1
                # print('-> [{}] generated from U[1,{}]'.format(rand_val, blank_duration_ms))
                output, offset_pos = overlap(output, audio_dict[i]['audio'], rand_val)
                targets[i]['offset'] = offset_pos

                # print(beg)
                # print(offset_pos)
                # print('___')
                # librosa.output.write_wav('toto_{}.wav'.format(i), output[beg:offset_pos], sr=self.sample_rate)

        # librosa.output.write_wav('toto.wav', output, sr=self.sample_rate)
        # print('buffer length = {}'.format(len(output)))
        # pprint.pprint(targets)
        return targets, output
