import os
import shutil
import time

from audio.audio_reader import AudioReader
from constants import c


def main():
    cache_output_dir = os.path.expanduser(c.AUDIO.CACHE_OUTPUT_PATH)
    print('The directory containing the cache is {}.'.format(cache_output_dir))
    print('Going to wipe out and regenerate the cache in 5 seconds. Ctrl+C to kill this script.')
    time.sleep(5)
    try:
        shutil.rmtree(cache_output_dir)
    except:
        pass
    os.makedirs(cache_output_dir)
    AudioReader(audio_dir=c.AUDIO.VCTK_CORPUS_PATH,
                sample_rate=c.AUDIO.SAMPLE_RATE,
                cache_output_dir=cache_output_dir,
                multi_threading_cache_generation=True,  # tested and both yield same result (single vs multi threads)
                speakers_sub_list=None)


if __name__ == '__main__':
    main()
