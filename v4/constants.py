LIBRI_SPEECH_DIR = 'audio/LibriSpeechSamples/'

NUM_FRAMES = 2
SAMPLE_RATE = 16000  # not higher than that.

TRAIN_TEST_RATIO = 0.8

CHECKPOINTS_DIR = 'checkpoints'
LOSS_FILE = CHECKPOINTS_DIR + '/losses.txt'
BATCH_SIZE = 20 * 3  # have to be a multiple of 3.
