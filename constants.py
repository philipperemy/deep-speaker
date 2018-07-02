DATASET_DIR = 'audio/LibriSpeechSamples/'

BATCH_NUM_TRIPLETS = 3  # should be a multiple of 3

# very dumb values. I selected them to have a blazing fast training.
# we will change them to their true values (to be defined?) later.
NUM_FRAMES = 2 # Not relevant anymore.

# https://wiki.audacityteam.org/wiki/Sample_Rates
SAMPLE_RATE = 8000  # Same as telephone audio sampled.

TRUNCATE_SOUND_SECONDS = (0.5, 1.5)  # (start_sec, end_sec)

CHECKPOINT_FOLDER = 'checkpoints'
LOSS_FILE = CHECKPOINT_FOLDER + '/losses.txt'
