import numpy as np


def get_mfcc_features_390(sig, rate, max_frames=None):
    window_length_sec = 25.0 / 1000
    window_step_sec = 10.0 / 1000
    window_cnn_fr_size = int(window_length_sec * rate)  # window size in frames
    window_cnn_fr_steps = int(window_step_sec * rate)  # the step size in frames. if step < window, overlap!
    feat_mat = []
    for i in range(int(len(sig) / window_cnn_fr_steps)):
        start = window_cnn_fr_steps * i
        end = start + window_cnn_fr_size
        slice_sig = sig[start:end]
        if len(slice_sig) / rate == window_length_sec:
            feat = mfcc_features(slice_sig, rate).flatten()
            feat_mat.append(feat)
    feat_mat = np.array(feat_mat, dtype=float)

    indices = np.array(range(10))
    new_feat_mat = []
    for frame_id in range(len(feat_mat)):
        if max(indices) >= len(feat_mat):
            break
        new_feat_mat.append(np.transpose(feat_mat[indices]).flatten())  # (39, 10).flatten()
        indices += 3
    new_feat_mat = np.array(new_feat_mat)
    if max_frames is not None:
        new_feat_mat = new_feat_mat[0:max_frames]
    return new_feat_mat


def mfcc_features(sig, rate, nb_features=13):
    from python_speech_features import mfcc, delta
    mfcc_feat = mfcc(sig, rate, numcep=nb_features, nfilt=nb_features)
    delta_feat = delta(mfcc_feat, 2)
    double_delta_feat = delta(delta_feat, 2)
    return np.concatenate((mfcc_feat, delta_feat, double_delta_feat), axis=1)


if __name__ == '__main__':
    signal = np.random.uniform(size=8000)
    get_mfcc_features_390(signal, 4000, max_frames=2)
