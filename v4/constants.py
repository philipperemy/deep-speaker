LIBRI_SPEECH_DIR = 'audio/LibriSpeechSamples/'


NUM_FRAMES = 2
SAMPLE_RATE = 16000  # not higher than that.

TRAIN_TEST_RATIO = 0.8

CHECKPOINT_FOLDER = 'checkpoints'
LOSS_FILE = CHECKPOINT_FOLDER + '/losses.txt'
BATCH_SIZE = 900