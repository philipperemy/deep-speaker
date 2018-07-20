import pickle
from glob import glob

import os
from keras import Model
from keras.layers import Dense
from natsort import natsorted

from constants import c
from ml.classifier_model_definition_triplet import get_model
from ml.classifier_model_definition_softmax import fit_model_softmax, build_model_softmax
from ml.utils import data_to_keras


def start_training():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    data_filename = '/tmp/speaker-change-detection-data.pkl'
    assert os.path.exists(data_filename), 'Data does not exist.'
    print('Loading the inputs in memory. It might take a while...')
    data = pickle.load(open(data_filename, 'rb'))
    kx_train, ky_train, kx_test, ky_test, categorical_speakers = data_to_keras(data)
    print('Dumping info about categorical speakers for the next phase (train distance classifier..')
    pickle.dump(categorical_speakers, open('/tmp/speaker-change-detection-categorical_speakers.pkl', 'wb'))
    print('Defining model...')
    num_speakers = c.AUDIO.NUM_SPEAKERS_CLASSIFICATION_TASK
    batch_size = 900
    m = get_model(batch_size)
    checkpoints = natsorted(glob('checkpoints/*.h5'))
    assert len(checkpoints) != 0, 'No checkpoints found.'
    checkpoint_file = checkpoints[-1]
    print('Loading checkpoint: {}.'.format(checkpoint_file))
    m.load_weights(checkpoint_file)  # latest one.
    print('Softmax size (num speakers) = {}'.format(num_speakers))
    top_layer = m.layers[-1].output
    top_layer = Dense(num_speakers, activation='softmax')(top_layer)
    m = Model(inputs=m.inputs, outputs=[top_layer])
    m.layers[1].trainable = False  # Big dense layer.
    # m.layers[2].trainable = False  # softmax.
    print(m.summary())
    build_model_softmax(m)  # categorical_crossentropy + adam
    print('Fitting softmax model...')
    fit_model_softmax(m, kx_train, ky_train, kx_test, ky_test, batch_size)


if __name__ == '__main__':
    start_training()
