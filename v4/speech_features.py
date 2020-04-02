import numpy as np
from python_speech_features import fbank, delta, mfcc


def normalize_frames(m, epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]


def get_mfcc_features_390(sig, rate, max_frames=None):
    def mfcc_features(sig_, rate_, nb_features_=13):
        mfcc_feat = mfcc(sig_, rate_, numcep=nb_features_, nfilt=nb_features_, nfft=1024)
        delta_feat = delta(mfcc_feat, 2)
        double_delta_feat = delta(delta_feat, 2)
        return np.concatenate((mfcc_feat, delta_feat, double_delta_feat), axis=1)

    window_length_sec = 25.0 / 1000
    window_step_sec = 10.0 / 1000
    window_cnn_fr_size = int(window_length_sec * rate)  # window size in frames
    window_cnn_fr_steps = int(window_step_sec * rate)  # the step size in frames. if step < window, overlap!
    feat_mat = []
    for i in range(int(((len(sig) - window_cnn_fr_size) / window_cnn_fr_steps))):
        start = window_cnn_fr_steps * i
        end = start + window_cnn_fr_size
        if end > len(sig):
            print('Hello.')
        slice_sig = sig[start:end]
        if len(slice_sig) / rate == window_length_sec:
            feat = mfcc_features(slice_sig, rate).flatten()
            feat_mat.append(feat)
        else:
            raise Exception('Sample rate_ probably too low.')
    feat_mat = np.array(feat_mat, dtype=float)

    indices = np.array(range(10))
    new_feat_mat = []
    for frame_id in range(len(feat_mat)):
        if max(indices) >= len(feat_mat):
            break
        new_feat_mat.append(np.transpose(feat_mat[indices]).flatten())
        indices += 3
    new_feat_mat = np.array(new_feat_mat)
    if max_frames is not None:
        new_feat_mat = new_feat_mat[0:max_frames]
    return new_feat_mat


def pre_process_inputs(sig=np.random.uniform(size=32000), target_sample_rate=8000):
    filter_banks, energies = fbank(sig, samplerate=target_sample_rate, nfilt=64)
    delta_1 = delta(filter_banks, N=1)
    delta_2 = delta(delta_1, N=1)
    # should not normalize here. That's an heresy!
    frames_features = np.transpose(np.stack([filter_banks, delta_1, delta_2]), (1, 2, 0))
    return frames_features


if __name__ == '__main__':
    pre_process_inputs()
    signal = np.random.uniform(size=8000)
    get_mfcc_features_390(signal, 4000, max_frames=2)
