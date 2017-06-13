import numpy as np
from python_speech_features import fbank, delta

from constants import *
from models import convolutional_model
from triplet_loss import deep_speaker_loss


def normalize_frames(m):
    return [(v - np.mean(v)) / np.std(v) for v in m]


if __name__ == '__main__':
    signal = np.random.uniform(size=4000)
    filter_banks, energies = fbank(signal, samplerate=4000, nfilt=64)
    delta_1 = delta(filter_banks, N=1)
    delta_2 = delta(delta_1, N=1)

    filter_banks = normalize_frames(filter_banks)
    delta_1 = normalize_frames(delta_1)
    delta_2 = normalize_frames(delta_2)

    frames_features = np.hstack([filter_banks, delta_1, delta_2, np.expand_dims(energies, axis=1)])
    num_frames = len(frames_features)
    network_inputs = []
    for j in range(8, num_frames - 8):
        frames_slice = frames_features[j - 8:j + 8]
        network_inputs.append(frames_slice)

    # TODO: wrong but just to make it work now.
    network_inputs = np.reshape(np.array(network_inputs)[0:BATCH_SIZE * NUM_FRAMES, 0:16, 0:16], (-1, 16, 16, 1))

    # model = convolutional_model(batch_input_shape=[BATCH_SIZE * NUM_FRAMES] + list(frames_slice.shape) + [1])
    model = convolutional_model(batch_input_shape=[BATCH_SIZE * NUM_FRAMES, 16, 16, 1])
    model.compile(optimizer='adam',
                  loss=deep_speaker_loss,
                  metrics=['accuracy'])

    print(model.summary())
    stub_targets = np.random.uniform(size=(BATCH_SIZE * NUM_FRAMES, 1))
    print(model.train_on_batch(network_inputs, stub_targets))

