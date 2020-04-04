import logging

import keras.backend as K
from keras import layers
from keras import regularizers
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Lambda, Dense
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from constants import BATCH_SIZE, NUM_FRAMES, NUM_FBANKS

logger = logging.getLogger(__name__)


class LayerCache:

    def __init__(self):
        self.layers_dict = dict()

    def get(self, obj):
        layer_name = obj.name
        if layer_name not in self.layers_dict:
            logger.info('-> Creating layer [{}]'.format(layer_name))
            # create it
            self.layers_dict[layer_name] = obj
        else:
            logger.info('-> Using layer [{}]'.format(layer_name))
        return self.layers_dict[layer_name]


class DeepSpeakerModel:

    def __init__(self, batch_input_shape=(BATCH_SIZE, NUM_FRAMES, NUM_FBANKS, 3)):
        batch_size = batch_input_shape[0]
        num_frames = batch_input_shape[1]
        self.lc = LayerCache()

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
        inputs = Input(batch_shape=batch_input_shape)
        x = self.cnn_component(inputs)
        x = Reshape((2048,))(x)
        x = Lambda(lambda y: K.reshape(y, (batch_size, num_frames, 2048)), name='reshape')(x)
        x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)
        x = Dense(512, name='affine')(x)  # .shape = (BATCH_SIZE * NUM_FRAMES, 512)
        x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
        self.m = Model(inputs, x, name='convolutional')

    def keras_model(self):
        return self.m

    # no need to share the weights here because it does not exist.
    def clipped_relu(self, inputs):
        return self.lc.get(Lambda(lambda y: K.minimum(K.maximum(y, 0), 20), name='clipped_relu'))(inputs)

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        conv_name_base = f'res{stage}_{block}_branch'

        x = self.lc.get(Conv2D(filters,
                               kernel_size=kernel_size,
                               strides=1,
                               activation=None,
                               padding='same',
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=regularizers.l2(l=0.0001),
                               name=conv_name_base + '_2a'))(input_tensor)
        x = self.lc.get(BatchNormalization(name=conv_name_base + '_2a_bn'))(x)
        x = self.clipped_relu(x)

        x = self.lc.get(Conv2D(filters,
                               kernel_size=kernel_size,
                               strides=1,
                               activation=None,
                               padding='same',
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=regularizers.l2(l=0.0001),
                               name=conv_name_base + '_2b'))(x)
        x = self.lc.get(BatchNormalization(name=conv_name_base + '_2b_bn'))(x)

        x = layers.add([x, input_tensor])
        x = self.clipped_relu(x)
        return x

    def conv_and_res_block(self, inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        o = self.lc.get(Conv2D(filters,
                               kernel_size=5,
                               strides=2,
                               padding='same',
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=regularizers.l2(l=0.0001), name=conv_name))(inp)
        o = self.lc.get(BatchNormalization(name=conv_name + '_bn'))(o)
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
