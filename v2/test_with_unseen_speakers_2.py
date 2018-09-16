import sys
from glob import glob

import numpy as np
from natsort import natsorted

from audio_reader import AudioReader
from constants import c
from speech_features import get_mfcc_features_390
from train_triplet_softmax_model import triplet_softmax_model
from utils import normalize, InputsGenerator


def get_feat_from_audio(audio_reader, sr, norm_data, speaker):
    feat = get_mfcc_features_390(audio_reader, sr, max_frames=None)
    feat = normalize(feat, norm_data[speaker]['mean_train'], norm_data[speaker]['std_train'])
    return feat


def generate_features_for_unseen_speakers(audio_reader, target_speaker='p363'):
    assert target_speaker in audio_reader.all_speaker_ids
    # audio.metadata = dict()  # small cache <SPEAKER_ID -> SENTENCE_ID, filename>
    # audio.cache = dict()  # big cache <filename, data:audio librosa, blanks.>

    inputs_generator = InputsGenerator(cache_dir=audio_reader.cache_dir,
                                       audio_reader=audio_reader,
                                       max_count_per_class=100)
    inputs = inputs_generator.generate_inputs(target_speaker)

    # audio_filename = list(audio_reader.metadata[target_speaker].values())[0]['filename']
    # audio_entity = audio_reader.cache[audio_filename]
    # feat = generate_features([audio_entity], 10)
    # feat = normalize(feat, inputs['mean_train'], inputs['std_train'])
    return inputs['test']


def test_on_unseen_speakers():
    assert len(sys.argv) == 3, 'Usage: [speaker_1] [speaker_2]. Example is p363 p362.'
    sp1 = sys.argv[1]
    sp2 = sys.argv[2]

    audio_reader = AudioReader(input_audio_dir=c.AUDIO.VCTK_CORPUS_PATH,
                               output_cache_dir=c.AUDIO.CACHE_PATH,
                               sample_rate=c.AUDIO.SAMPLE_RATE)

    sp1_feat = generate_features_for_unseen_speakers(audio_reader, target_speaker=sp1)
    sp2_feat = generate_features_for_unseen_speakers(audio_reader, target_speaker=sp2)

    # batch_size => None (for inference).
    m = triplet_softmax_model(num_speakers_softmax=len(c.AUDIO.SPEAKERS_TRAINING_SET),
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

    emb_sp1 = m.predict(np.vstack(sp1_feat))[0]
    emb_sp2 = m.predict(np.vstack(sp2_feat))[0]

    print('Checking that L2 norm is 1.')
    print(np.mean(np.linalg.norm(emb_sp1, axis=1)))
    print(np.mean(np.linalg.norm(emb_sp2, axis=1)))

    from scipy.spatial.distance import cosine

    # note to myself:
    # embeddings are sigmoid-ed.
    # so they are between 0 and 1.
    # A hypersphere is defined on tanh.

    print('Emb1 = ', emb_sp1.shape)
    print('Emb2 = ', emb_sp2.shape)

    emb1 = np.mean(emb_sp1, axis=0)
    emb2 = np.mean(emb_sp2, axis=0)

    print('Cosine =', cosine(emb1, emb2))

    # print('SAP =', np.mean([cosine(u, v) for (u, v) in zip(emb_sp1[:-1], emb_sp1[1:])]))
    # print('SAN =', np.mean([cosine(u, v) for (u, v) in zip(emb_sp1, emb_sp2)]))
    # print('We expect: SAP << SAN.')


if __name__ == '__main__':
    test_on_unseen_speakers()
