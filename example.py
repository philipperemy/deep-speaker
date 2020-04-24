import numpy as np
import random
from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from test import batch_cosine_similarity

np.random.seed(123)
random.seed(123)

model = DeepSpeakerModel()
model.m.load_weights('/Users/premy/deep-speaker/checkpoints/ResCNN_triplet_training_checkpoint_175.h5', by_name=True)

mfcc_001 = sample_from_mfcc(read_mfcc('samples/PhilippeRemy/PhilippeRemy_001.wav', SAMPLE_RATE), NUM_FRAMES)
mfcc_002 = sample_from_mfcc(read_mfcc('samples/PhilippeRemy/PhilippeRemy_002.wav', SAMPLE_RATE), NUM_FRAMES)

predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))

mfcc_003 = sample_from_mfcc(read_mfcc('samples/1255-90413-0001.flac', SAMPLE_RATE), NUM_FRAMES)
predict_003 = model.m.predict(np.expand_dims(mfcc_003, axis=0))

print('SAME SPEAKER', batch_cosine_similarity(predict_001, predict_002))
print('DIFF SPEAKER', batch_cosine_similarity(predict_001, predict_003))
