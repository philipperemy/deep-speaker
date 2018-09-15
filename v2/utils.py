import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm

from audio.audio_reader import AudioReader, extract_speaker_id
from audio.speech_features import get_mfcc_features_390
from constants import c


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


def generate_features(audio_entities, max_count, desc):
    features = []
    for _ in tqdm(range(max_count), desc=desc):
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


def generate_inputs(max_count_per_class=500):
    output = dict()
    normalization_constants = dict()
    audio = AudioReader(audio_dir=c.AUDIO.VCTK_CORPUS_PATH,
                        sample_rate=c.AUDIO.SAMPLE_RATE,
                        cache_dir=c.AUDIO.CACHE_PATH)
    print(audio.get_speaker_list())

    per_speaker_dict = dict()
    for filename, audio_entity in audio.cache.items():
        speaker_id = extract_speaker_id(audio_entity['filename'])
        if speaker_id not in per_speaker_dict:
            per_speaker_dict[speaker_id] = []
        per_speaker_dict[speaker_id].append(audio_entity)

    for speaker_id, audio_entities in per_speaker_dict.items():
        print('Processing speaker id = {}.'.format(speaker_id))
        cutoff = int(len(audio_entities) * 0.8)
        audio_entities_train = audio_entities[0:cutoff]
        audio_entities_test = audio_entities[cutoff:]

        train = generate_features(audio_entities_train, max_count_per_class, 'train')
        print('Processed {} for train/'.format(max_count_per_class))
        test = generate_features(audio_entities_test, max_count_per_class, 'test')
        print('Processed {} for test/'.format(max_count_per_class))

        mean_train = np.mean([np.mean(t) for t in train])
        std_train = np.mean([np.std(t) for t in train])

        train = normalize(train, mean_train, std_train)
        test = normalize(test, mean_train, std_train)

        if speaker_id in c.AUDIO.SPEAKERS_TRAINING_SET:
            inputs = {'train': train, 'test': test, 'speaker_id': speaker_id}
            output[speaker_id] = inputs
            print('Adding speaker to the classification dataset: {}'.format(speaker_id))
        else:
            print('Discarding speaker for the classification dataset: {}'.format(speaker_id))
        # still we want to normalize all the speakers.
        normalization_constants[speaker_id] = {'mean_train': mean_train, 'std_train': std_train}
    return output, normalization_constants


class SpeakersToCategorical:
    def __init__(self, data):
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