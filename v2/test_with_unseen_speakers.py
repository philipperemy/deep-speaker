import json
import pickle
from glob import glob

import numpy as np
from natsort import natsorted

from audio_reader import AudioReader
from constants import c
from speech_features import get_mfcc_features_390
from train_triplet_softmax_model import triplet_softmax_model
from utils import generate_features, normalize

audio = AudioReader(input_audio_dir=c.AUDIO.VCTK_CORPUS_PATH,
                    output_cache_dir=c.AUDIO.CACHE_PATH,
                    sample_rate=c.AUDIO.SAMPLE_RATE)


def get_feat_from_audio(audio, sr, norm_data, speaker):
    feat = get_mfcc_features_390(audio, sr, max_frames=None)
    feat = normalize(feat, norm_data[speaker]['mean_train'], norm_data[speaker]['std_train'])
    return feat


def generate_features_for_unseen_speakers(norm_data, target_speaker='p363'):
    assert target_speaker in audio.get_speaker_list()
    # audio.metadata = dict()  # small cache <SPEAKER_ID -> SENTENCE_ID, filename>
    # audio.cache = dict()  # big cache <filename, data:audio librosa, blanks.>
    audio_filename = list(audio.metadata[target_speaker].values())[0]['filename']
    audio_entity = audio.cache[audio_filename]
    feat = generate_features([audio_entity], 10)
    feat = normalize(feat, norm_data[target_speaker]['mean_train'], norm_data[target_speaker]['std_train'])
    return feat


def test_on_unseen_speakers():
    categorical_speakers = pickle.load(open('/tmp/speaker-change-detection-categorical_speakers.pkl', 'rb'))
    norm_data = json.load(open('/tmp/speaker-change-detection-norm.json', 'r'))

    p363_feat = generate_features_for_unseen_speakers(norm_data, target_speaker='p363')
    p362_feat = generate_features_for_unseen_speakers(norm_data, target_speaker='p362')

    # batch_size => None (for inference).
    m = triplet_softmax_model(num_speakers_softmax=len(categorical_speakers.speaker_ids),
                              emb_trainable=False,
                              normalize_embeddings=True,
                              batch_size=None)

    checkpoints = natsorted(glob('checkpoints/*.h5'))

    # compile_triplet_softmax_model(m, loss_on_softmax=False, loss_on_embeddings=False)
    print(m.summary())

    if len(checkpoints) != 0:
        checkpoint_file = checkpoints[-1]
        initial_epoch = int(checkpoint_file.split('/')[-1].split('.')[0].split('_')[-1])
        print('Initial epoch is {}.'.format(initial_epoch))
        print('Loading checkpoint: {}.'.format(checkpoint_file))
        m.load_weights(checkpoint_file)  # latest one.

    emb_p363 = m.predict(np.vstack(p363_feat))[0]
    emb_p362 = m.predict(np.vstack(p362_feat))[0]

    print('Checking that L2 norm is 1.')
    print(np.mean(np.linalg.norm(emb_p363, axis=1)))
    print(np.mean(np.linalg.norm(emb_p362, axis=1)))

    from scipy.spatial.distance import cosine

    # note to myself:
    # embeddings are sigmoid-ed.
    # so they are between 0 and 1.
    # A hypersphere is defined on tanh.

    print('SAP =', np.mean([cosine(u, v) for (u, v) in zip(emb_p363[:-1], emb_p363[1:])]))
    print('SAN =', np.mean([cosine(u, v) for (u, v) in zip(emb_p363, emb_p362)]))
    print('We expect: SAP << SAN.')


if __name__ == '__main__':
    test_on_unseen_speakers()
