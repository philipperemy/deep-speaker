import abc
import logging

import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model

from constants import NUM_FBANKS, NUM_FRAMES

logger = logging.getLogger(__name__)

RES_CNN_NAME = 'ResCNN'
GRU_NAME = 'GRU'


def select_model_class(name: str):
    if name == RES_CNN_NAME:
        return ResCNNModel
    elif name == GRU_NAME:
        return GRUModel
    else:
        raise Exception(f'Unknown model name: {name}.')


class DeepSpeakerModel:

    def __init__(self,
                 batch_input_shape=(None, NUM_FRAMES, NUM_FBANKS, 1),
                 include_softmax=False,
                 num_speakers_softmax=None,
                 name=RES_CNN_NAME):
        self.include_softmax = include_softmax
        self.num_speakers_softmax = num_speakers_softmax
        if self.include_softmax:
            assert self.num_speakers_softmax > 0
        self.clipped_relu_count = 0
        inputs = Input(batch_shape=batch_input_shape, name='input')
        x = self.graph_with_avg_softmax_and_ln(inputs)
        self.m = Model(inputs, x, name=name)

    @abc.abstractmethod
    def graph(self, inputs):
        pass

    def graph_with_avg_softmax_and_ln(self, inputs):
        x = self.graph(inputs)
        # Temporal average layer. axis=1 is time.
        x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)
        if self.include_softmax:
            logger.info('Including a Dropout layer to reduce overfitting.')
            # used for softmax because the dataset we pre-train on might be too small. easy to overfit.
            # x = Dropout(0.25)(x) # was for GRU. Does 0.5 work with GRU as well?
            x = Dropout(0.5)(x)
        x = Dense(512, name='affine')(x)
        if self.include_softmax:
            # Those weights are just when we train on softmax.
            x = Dense(self.num_speakers_softmax, activation='softmax')(x)
        else:
            # Does not contain any weights.
            x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
        return x

    def keras_model(self):
        return self.m

    def get_weights(self):
        w = self.m.get_weights()
        if self.include_softmax:
            w.pop()  # last 2 are the W_softmax and b_softmax.
            w.pop()
        return w

    def clipped_relu(self, inputs):
        relu = Lambda(lambda y: K.minimum(K.maximum(y, 0), 20), name=f'clipped_relu_{self.clipped_relu_count}')(inputs)
        self.clipped_relu_count += 1
        return relu

    def set_weights(self, w):
        for layer, layer_w in zip(self.m.layers, w):
            layer.set_weights(layer_w)
            logger.info(f'Setting weights for [{layer.name}]...')


class ResCNNModel(DeepSpeakerModel):

    def __init__(self,
                 batch_input_shape=(None, NUM_FRAMES, NUM_FBANKS, 1),
                 include_softmax=False,
                 num_speakers_softmax=None):
        super().__init__(batch_input_shape, include_softmax, num_speakers_softmax, RES_CNN_NAME)

    def graph(self, inputs):
        x = self.conv_and_res_block(inputs, 64, stage=1)
        x = self.conv_and_res_block(x, 128, stage=2)
        x = self.conv_and_res_block(x, 256, stage=3)
        x = self.conv_and_res_block(x, 512, stage=4)
        x = Reshape((-1, 2048))(x)
        return x

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


class GRUModel(DeepSpeakerModel):

    def __init__(self,
                 batch_input_shape=(None, NUM_FRAMES, NUM_FBANKS, 1),
                 include_softmax=False,
                 num_speakers_softmax=None):
        super().__init__(batch_input_shape, include_softmax, num_speakers_softmax, GRU_NAME)

    def graph(self, inputs):
        x = Conv2D(64, kernel_size=5, strides=2, padding='same', kernel_initializer='glorot_uniform',
                   name='conv1', kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
        # shape = (BATCH_SIZE , num_frames/2, 64/2, 64)
        x = BatchNormalization(name='bn1')(x) # does it work with BN?
        x = self.clipped_relu(x)

        # 4d -> 3d.
        _, frames_dim, fbank_dim, conv_output_dim = K.int_shape(x)
        x = Reshape((frames_dim, fbank_dim * conv_output_dim))(x)
        x = Reshape((frames_dim, fbank_dim * conv_output_dim))(x)

        # shape = (BATCH_SIZE , num_frames/2, 1024)
        x = GRU(1024, name='GRU1', return_sequences=True)(x)
        if self.include_softmax:
            x = Dropout(0.2)(x)
        x = GRU(1024, name='GRU2', return_sequences=True)(x)
        if self.include_softmax:
            x = Dropout(0.2)(x)
        x = GRU(1024, name='GRU3', return_sequences=True)(x)
        return x

