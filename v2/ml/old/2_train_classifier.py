import pickle

import os

from ml.classifier_model_definition_triplet import get_model, fit_model, build_model
from ml.utils import data_to_keras


def start_training():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # num_speakers = c.AUDIO.NUM_SPEAKERS_CLASSIFICATION_TASK
    data_filename = '/tmp/speaker-change-detection-data.pkl'
    assert os.path.exists(data_filename), 'Data does not exist.'
    print('Loading the inputs in memory. It might take a while...')
    data = pickle.load(open(data_filename, 'rb'))
    kx_train, ky_train, kx_test, ky_test, categorical_speakers = data_to_keras(data)
    print('Dumping info about categorical speakers for the next phase (train distance classifier)..')
    pickle.dump(categorical_speakers, open('/tmp/speaker-change-detection-categorical_speakers.pkl', 'wb'))
    print('Defining model...')
    batch_size = 900
    m = get_model(batch_size)
    print('Building model...')
    build_model(m)
    print('Fitting model...')
    fit_model(m, kx_train, ky_train, kx_test, ky_test, batch_size=batch_size)


if __name__ == '__main__':
    start_training()
