# Constants.

SAMPLE_RATE = 16000  # not higher than that otherwise we may have errors when computing the fbanks.

# Train/Test sets share the same speakers. They contain different utterances.
TRAIN_TEST_RATIO = 0.8

CHECKPOINTS_DIR = 'checkpoints'

BATCH_SIZE = 20 * 3  # have to be a multiple of 3.

# Input to the model will be a 4D image: (batch_size, num_frames, num_fbanks, 3)
# Where the 3 channels are: FBANK, DIFF(FBANK), DIFF(DIFF(FBANK)).
NUM_FRAMES = 28
NUM_FBANKS = 64
