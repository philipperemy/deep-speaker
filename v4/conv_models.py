import logging

import keras.backend as K
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from constants import NUM_FBANKS, NUM_FRAMES
from triplet_loss import deep_speaker_loss

logger = logging.getLogger(__name__)


class DeepSpeakerModel:

    # I thought it was 3 but maybe energy is added at a 4th dimension.
    # would be better to have 4 dimensions:
    # MFCC, DIFF(MFCC), DIFF(DIFF(MFCC)), ENERGIES (probably tiled across the frequency domain).
    # this seems to help match the parameter counts.
    def __init__(self, batch_input_shape=(None, NUM_FRAMES, NUM_FBANKS, 4), include_softmax=False,
                 num_speaker_softmax=None):
        if include_softmax:
            assert len(num_speaker_softmax) > 0
        self.clipped_relu_count = 0

        # http://cs231n.github.io/convolutional-networks/
        # conv weights
        # #params = ks * ks * nb_filters * num_channels_input

        # Conv128-s
        # 5*5*128*128/2+128
        # ks*ks*nb_filters*channels/strides+bias(=nb_filters)

        # take 100 ms -> 4 frames.
        # if signal is 3 seconds, then take 100ms per 100ms and average out this network.
        # 8*8 = 64 features.

        # used to share all the layers across the inputs

        # num_frames = K.shape() - do it dynamically after.
        inputs = Input(batch_shape=batch_input_shape, name='input')
        x = self.cnn_component(inputs)
        # TODO: not sure about this. But one thing for sure is that any result of a Conv will be a 4D shape.
        # TODO: it's either this or run:
        # Flatten() and FC()-it, re-run the model several times. And averages to have it at utterance level.
        # One way to do that is to pass everything in the batch dim, run and reshape.
        x = Reshape((-1, 2048))(x)
        x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)
        x = Dense(512, name='affine')(x)  # .shape = (BATCH_SIZE * NUM_FRAMES, 512)
        if include_softmax:
            x = Dense(num_speaker_softmax, activation='softmax')(x)
        else:
            x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
        self.m = Model(inputs, x, name='ResCNN')

    def keras_model(self):
        return self.m

    def clipped_relu(self, inputs):
        relu = Lambda(lambda y: K.minimum(K.maximum(y, 0), 20), name=f'clipped_relu_{self.clipped_relu_count}')(inputs)
        self.clipped_relu_count += 1
        return relu

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        conv_name_base = f'res{stage}_{block}_branch'

        x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001),
                   name=conv_name_base + '_2a')(input_tensor)
        x = BatchNormalization(name=conv_name_base + '_2a_bn')(x)
        x = self.clipped_relu(x)

        x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001),
                   name=conv_name_base + '_2b')(x)
        x = BatchNormalization(name=conv_name_base + '_2b_bn')(x)

        x = self.clipped_relu(x)

        x = layers.add([x, input_tensor])
        x = self.clipped_relu(x)
        return x

    def conv_and_res_block(self, inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        # TODO: why kernel_regularizer?
        o = Conv2D(filters,
                   kernel_size=5,
                   strides=2,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001), name=conv_name)(inp)
        o = BatchNormalization(name=conv_name + '_bn')(o)
        o = self.clipped_relu(o)
        for i in range(3):
            o = self.identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
        return o

    def cnn_component(self, inp):
        x = self.conv_and_res_block(inp, 64, stage=1)
        x = self.conv_and_res_block(x, 128, stage=2)
        x = self.conv_and_res_block(x, 256, stage=3)
        x = self.conv_and_res_block(x, 512, stage=4)
        return x


def main():
    # Looks correct to me.
    # I have 37K but paper reports 41K. which is not too far.
    dsm = DeepSpeakerModel()
    dsm.m.summary()

    # I suspect num frames to be 32.
    # Then fbank=64, then total would be 32*64 = 2048.
    # plot_model(dsm.m, to_file='model.png', dpi=300, show_shapes=True, expand_nested=True)


def train():
    dsm = DeepSpeakerModel()
    dsm.m.summary()
    dsm.m.compile(optimizer=Adam(lr=0.0001), loss=deep_speaker_loss)
    x = np.random.uniform(size=(6, 32, 64, 4))  # 6 is multiple of 3.
    # should be easy to learn this.
    x[0:2] = -0.1  # anchor
    x[2:4] = x[0:2]  # positive
    x[4:6] = 0.1  # negative
    y = np.zeros(shape=(6, 512))  # not important.
    print(dsm.m.evaluate(x, y))


if __name__ == '__main__':
    main()
