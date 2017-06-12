import numpy as np
from python_speech_features import fbank, delta

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
    for j in range(8, num_frames - 7):
        frames_slice = frames_features[j - 8:j + 7]
        network_inputs.append(frames_slice)

    model = convolutional_model(input_shapes=list(network_inputs[0].shape) + [1],
                                num_frames=len(network_inputs[0]))
    model.compile(optimizer='adam',
                  loss=deep_speaker_loss,
                  metrics=['accuracy'])

    model.fit(network_inputs, np.array([0] * len(network_inputs)))

    print(model.summary())
