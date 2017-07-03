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

from librispeech_wav_reader import read_audio


class MiniBatch:
    def __init__(self, libri_batch):
        self.libri_batch = libri_batch
        self.audio_loaded = False

    def load_wav(self):
        self.libri_batch['raw_audio'] = self.libri_batch['filename'].apply(lambda x: read_audio(x)[0])
        self.audio_loaded = True

    def to_inputs(self):
        x = self.libri_batch['raw_audio'].values
        y = self.libri_batch['speaker_id'].values
        return x, y


def stochastic_mini_batch(libri, batch_size):
    # indices = np.random.choice(len(libri), size=batch_size, replace=False)
    libri_batch = libri.sample(n=batch_size, replace=False)
    mini_batch = MiniBatch(libri_batch)
    mini_batch.load_wav()
    mini_batch.to_inputs()
    a = 2


def test():
    from librispeech_wav_reader import read_librispeech_structure
    libri = read_librispeech_structure('/Volumes/Transcend/data-set/LibriSpeech')
    stochastic_mini_batch(libri, 32)


if __name__ == '__main__':
    test()
