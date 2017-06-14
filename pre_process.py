import numpy as np
from python_speech_features import fbank, delta


def normalize_frames(m):
    return [(v - np.mean(v)) / np.std(v) for v in m]


def next_batch():
    signal = np.random.uniform(size=32000)  # 4 seconds
    filter_banks, energies = fbank(signal, samplerate=8000, nfilt=64, winlen=0.025)
    delta_1 = delta(filter_banks, N=1)
    delta_2 = delta(delta_1, N=1)

    filter_banks = normalize_frames(filter_banks)
    delta_1 = normalize_frames(delta_1)
    delta_2 = normalize_frames(delta_2)

    frames_features = np.hstack([filter_banks, delta_1, delta_2])
    num_frames = len(frames_features)
    network_inputs = []
    for j in range(8, num_frames - 8):
        frames_slice = frames_features[j - 8:j + 8]
        network_inputs.append(np.reshape(frames_slice, (32, 32, 3)))
    return np.array(network_inputs)


if __name__ == '__main__':
    b = next_batch()
    print(b.shape)
