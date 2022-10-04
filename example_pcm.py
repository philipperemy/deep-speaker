import librosa
import numpy as np
import tensorflow as tf

from deep_speaker.constants import SAMPLE_RATE
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity

# Define the model here.
model = DeepSpeakerModel(pcm_input=True)

# Load the checkpoint.
model.m.load_weights('ResCNN_triplet_training_checkpoint_265.h5', by_name=True)

samples = [
    'samples/PhilippeRemy/PhilippeRemy_001.wav',
    'samples/PhilippeRemy/PhilippeRemy_002.wav',
    'samples/1255-90413-0001.flac',
]

pcm = [librosa.load(x, sr=SAMPLE_RATE, mono=True)[0] for x in samples]

# Crop samples in the center, to fit the smaller audio samples
num_samples = min([len(x) for x in pcm])
pcm = tf.convert_to_tensor(np.stack([x[(len(x) - num_samples) // 2:][:num_samples] for x in pcm]))
# Call the model to get the embeddings of shape (1, 512) for each file.
predict = model.m.predict(pcm)
speaker_similarity = batch_cosine_similarity(predict[0:1], predict[1:])

# Compute the cosine similarity and check that it is higher for the same speaker.
same_speaker_similarity = speaker_similarity[0]
diff_speaker_similarity = speaker_similarity[1]
print('SAME SPEAKER', same_speaker_similarity)  # SAME SPEAKER [0.81564593]
print('DIFF SPEAKER', diff_speaker_similarity)  # DIFF SPEAKER [0.1419204]

assert same_speaker_similarity > diff_speaker_similarity
