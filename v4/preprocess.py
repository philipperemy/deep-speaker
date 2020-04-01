import numpy as np
# def normalize_frames(m):
#    return [(v - np.mean(v)) / np.std(v) for v in m]
import python_speech_features

from audio import read_audio
from constants import TRUNCATE_SOUND_SECONDS, SAMPLE_RATE


def normalize_frames(m, epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]


def load_wav(libri):
    libri['raw_audio'] = libri['filename'].apply(lambda x: read_audio(x))
    min_existing_frames = min(libri['raw_audio'].apply(lambda x: len(x)).values)
    start_sec, end_sec = TRUNCATE_SOUND_SECONDS
    start_frame = int(start_sec * SAMPLE_RATE)
    end_frame = min(int(end_sec * SAMPLE_RATE), min_existing_frames)
    libri['raw_audio'] = libri['raw_audio'].apply(lambda x: x[start_frame:end_frame])
    return libri


def pre_process_inputs(signal=np.random.uniform(size=32000), target_sample_rate=8000):
    filter_banks, energies = python_speech_features.fbank(signal, samplerate=target_sample_rate, nfilt=64, winlen=0.025)
    delta_1 = python_speech_features.delta(filter_banks, N=1)
    delta_2 = python_speech_features.delta(delta_1, N=1)

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
