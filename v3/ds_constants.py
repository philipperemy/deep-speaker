DATASET_DIR = 'audio/LibriSpeechSamples/'

BATCH_NUM_TRIPLETS = 6  # should be a multiple of 3

# very dumb values. I selected them to have a blazing fast training.
# we will change them to their true values (to be defined?) later.
NUM_FRAMES = 2
SAMPLE_RATE = 100
TRUNCATE_SOUND_SECONDS = (0.5,1.5)  # (start_sec, end_sec)

CHECKPOINT_FOLDER = 'checkpoints'
LOSS_FILE = CHECKPOINT_FOLDER + '/losses.txt'
