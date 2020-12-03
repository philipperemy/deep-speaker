import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

from constants import NUM_FRAMES, NUM_FBANKS
from models import ResCNNModel, GRUModel
from triplet_loss import deep_speaker_loss


def _train():
    # x = np.random.uniform(size=(6, 32, 64, 4))  # 6 is multiple of 3.
    # y_softmax = np.random.uniform(size=(6, 100))
    # dsm = DeepSpeakerModel(batch_input_shape=(None, 32, 64, 4), include_softmax=True, num_speakers_softmax=100)
    # dsm.m.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy')
    # print(dsm.m.predict(x).shape)
    # print(dsm.m.evaluate(x, y_softmax))
    # w = dsm.get_weights()
    dsm = ResCNNModel(batch_input_shape=(None, 32, 64, 4), include_softmax=False)
    # dsm.m.set_weights(w)
    dsm.m.compile(optimizer=Adam(lr=0.01), loss=deep_speaker_loss)

    # it works!!!!!!!!!!!!!!!!!!!!
    # unit_batch_size = 20
    # anchor = np.ones(shape=(unit_batch_size, 32, 64, 4))
    # positive = np.array(anchor)
    # negative = np.ones(shape=(unit_batch_size, 32, 64, 4)) * (-1)
    # batch = np.vstack((anchor, positive, negative))
    # x = batch
    # y = np.zeros(shape=(len(batch), 512))  # not important.
    # print('Starting to fit...')
    # while True:
    #     print(dsm.m.train_on_batch(x, y))

    # should not work... and it does not work!
    unit_batch_size = 20
    negative = np.ones(shape=(unit_batch_size, 32, 64, 4)) * (-1)
    batch = np.vstack((negative, negative, negative))
    x = batch
    y = np.zeros(shape=(len(batch), 512))  # not important.
    print('Starting to fit...')
    while True:
        print(dsm.m.train_on_batch(x, y))


def test_gru():
    # Looks correct to me.
    # I have 37K but paper reports 41K. which is not too far.
    dsm = GRUModel()
    dsm.m.summary()
    dsm.m.predict(np.random.random(size=(2, NUM_FRAMES, NUM_FBANKS, 1)))

    i = np.random.random(size=(2, NUM_FRAMES, NUM_FBANKS, 1))
    y = K.reshape(i, (2, NUM_FRAMES, NUM_FBANKS * 1))
    z = K.reshape(y, (2, NUM_FRAMES, NUM_FBANKS, 1))

    # I suspect num frames to be 32.
    # Then fbank=64, then total would be 32*64 = 2048.
    # plot_model(dsm.m, to_file='model.png', dpi=300, show_shapes=True, expand_nested=True)
