import pickle
from glob import glob

import numpy as np
from natsort import natsorted

from audio.audio_reader import AudioReader
from audio.speech_features import get_mfcc_features_390
from constants import c
from ml.classifier_data_generation import normalize, generate_features
from ml.train_triplet_softmax_model import triplet_softmax_model

audio = AudioReader(audio_dir=c.AUDIO.VCTK_CORPUS_PATH,
                    sample_rate=c.AUDIO.SAMPLE_RATE,
                    speakers_sub_list=None)


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
    feat = generate_features([audio_entity], 10, 'generate_features')
    feat = normalize(feat, norm_data[target_speaker]['mean_train'], norm_data[target_speaker]['std_train'])
    return feat


def test_on_unseen_speakers():
    # change that later.
    # import os
    # data_filename = '/tmp/speaker-change-detection-data.pkl'
    # assert os.path.exists(data_filename), 'Data does not exist.'
    # print('Loading the inputs in memory. It might take a while...')
    # data = pickle.load(open(data_filename, 'rb'))
    # kx_train, ky_train, kx_test, ky_test, categorical_speakers = data_to_keras(data)
    # print('Dumping info about categorical speakers for the next phase (train distance classifier..')
    # pickle.dump(categorical_speakers, open('/tmp/speaker-change-detection-categorical_speakers.pkl', 'wb'))

    categorical_speakers = pickle.load(open('/tmp/speaker-change-detection-categorical_speakers.pkl', 'rb'))
    norm_data = pickle.load(open('/tmp/speaker-change-detection-norm.pkl', 'rb'))

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
    from sklearn.metrics.pairwise import cosine_similarity

    # note to myself:
    # embeddings are sigmoid-ed.
    # so they are between 0 and 1.
    # A hypersphere is defined on tanh.

    print('SAP =', np.mean([cosine(u, v) for (u, v) in zip(emb_p363[:-1], emb_p363[1:])]))
    print('SAN =', np.mean([cosine(u, v) for (u, v) in zip(emb_p363, emb_p362)]))
    print('We expect: SAP << SAN.')


if __name__ == '__main__':
    test_on_unseen_speakers()
