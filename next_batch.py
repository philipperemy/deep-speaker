"""
   filename                             chapter_id speaker_id dataset_id
0  1272/128104/1272-128104-0000.wav     128104       1272  dev-clean
1  1272/128104/1272-128104-0001.wav     128104       1272  dev-clean
2  1272/128104/1272-128104-0002.wav     128104       1272  dev-clean
3  1272/128104/1272-128104-0003.wav     128104       1272  dev-clean
4  1272/128104/1272-128104-0004.wav     128104       1272  dev-clean
5  1272/128104/1272-128104-0005.wav     128104       1272  dev-clean
6  1272/128104/1272-128104-0006.wav     128104       1272  dev-clean
7  1272/128104/1272-128104-0007.wav     128104       1272  dev-clean
8  1272/128104/1272-128104-0008.wav     128104       1272  dev-clean
9  1272/128104/1272-128104-0009.wav     128104       1272  dev-clean
"""

import numpy as np
import pandas as pd

import constants as c
from librispeech_wav_reader import read_audio


class MiniBatch:
    def __init__(self, libri_batch):
        self.libri_batch = libri_batch
        self.audio_loaded = False

    def load_wav(self):
        self.libri_batch['raw_audio'] = self.libri_batch['filename'].apply(lambda x: read_audio(x))
        min_existing_frames = min(self.libri_batch['raw_audio'].apply(lambda x: len(x)).values)
        max_frames = min(c.TRUNCATE_SOUND_FIRST_SECONDS * c.SAMPLE_RATE, min_existing_frames)
        self.libri_batch['raw_audio'] = self.libri_batch['raw_audio'].apply(lambda x: x[0:max_frames])
        self.audio_loaded = True

    def to_inputs(self):
        x = self.libri_batch['raw_audio'].values
        x = np.array([a.flatten() for a in x])
        y = self.libri_batch['speaker_id'].values
        return x, y


def stochastic_mini_batch(libri, batch_size):
    # indices = np.random.choice(len(libri), size=batch_size, replace=False)
    # [anc1, anc2, anc3, pos1, pos2, pos3, neg1, neg2, neg3]
    # [sp1, sp2, sp3, sp1, sp2, sp3, sp4, sp5, sp6]

    unique_speakers = list(libri['speaker_id'].unique())
    num_triplets = batch_size // 3

    anchor_batch = None
    positive_batch = None
    negative_batch = None
    for ii in range(num_triplets):
        two_different_speakers = np.random.choice(unique_speakers, size=2, replace=False)
        anchor_positive_speaker = two_different_speakers[0]
        negative_speaker = two_different_speakers[1]
        anchor_positive_file = libri[libri['speaker_id'] == anchor_positive_speaker].sample(n=2, replace=False)
        anchor_df = pd.DataFrame(anchor_positive_file[0:1])
        anchor_df['training_type'] = 'anchor'
        positive_df = pd.DataFrame(anchor_positive_file[1:2])
        positive_df['training_type'] = 'positive'
        negative_df = libri[libri['speaker_id'] == negative_speaker].sample(n=1)
        negative_df['training_type'] = 'negative'

        if anchor_batch is None:
            anchor_batch = anchor_df.copy()
        else:
            anchor_batch = pd.concat([anchor_batch, anchor_df], axis=0)

        if positive_batch is None:
            positive_batch = positive_df.copy()
        else:
            positive_batch = pd.concat([positive_batch, positive_df], axis=0)

        if negative_batch is None:
            negative_batch = negative_df.copy()
        else:
            negative_batch = pd.concat([negative_batch, negative_df], axis=0)

    libri_batch = pd.concat([anchor_batch, positive_batch, negative_batch], axis=0)
    mini_batch = MiniBatch(libri_batch)
    mini_batch.load_wav()
    mini_batch.to_inputs()



def test():
    from librispeech_wav_reader import read_librispeech_structure
    libri = read_librispeech_structure('/Volumes/Transcend/data-set/LibriSpeech')
    stochastic_mini_batch(libri, 32)


if __name__ == '__main__':
    test()
