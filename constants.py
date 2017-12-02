DATASET_DIR = 'audio/LibriSpeechSamples/'

BATCH_NUM_TRIPLETS = 6  # should be a multiple of 3

# very dumb values. I selected them to have a blazing fast training.
# we will change them to their true values (to be defined?) later.
NUM_FRAMES = 2
SAMPLE_RATE = 100
TRUNCATE_SOUND_FIRST_SECONDS = 1

CHECKPOINT_FOLDER = 'checkpoints'
