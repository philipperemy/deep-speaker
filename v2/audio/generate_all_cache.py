

from audio.audio_reader import AudioReader
from constants import c


def main():
    speakers_sub_list = ['p225', 'p226', 'p227', 'p228', 'p312', 'p313', 'p314', 'p315', 'p316']
    # speakers_sub_list = None
    AudioReader(audio_dir=c.AUDIO.VCTK_CORPUS_PATH,
                sample_rate=c.AUDIO.SAMPLE_RATE,
                speakers_sub_list=speakers_sub_list)


if __name__ == '__main__':
    main()
