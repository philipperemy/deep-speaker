import keras.backend as K
from keras import layers
from keras import regularizers
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def clipped_relu(inputs):
    return Lambda(lambda y: K.minimum(K.maximum(y, 0), 20))(inputs)


def identity_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res{}_{}_branch'.format(stage, block)

    x = Conv2D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001),
               name=conv_name_base + '_2a')(input_tensor)
    x = BatchNormalization(name=conv_name_base + '_2a_bn')(x)
    # x = Activation('relu')(x)
    x = clipped_relu(x)

    x = Conv2D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001),
               name=conv_name_base + '_2b')(x)
    x = BatchNormalization(name=conv_name_base + '_2b_bn')(x)

    # 1x1 conv
    x = Conv2D(filters,
               kernel_size=1,
               strides=1,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001),
               name=conv_name_base + '2c_bn')(x)

    x = layers.add([x, input_tensor])

    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = clipped_relu(x)
    return x


def convolutional_model(num_frames=4):
    # http://cs231n.github.io/convolutional-networks/
    # conv weights
    # #params = ks * ks * nb_filters * num_channels_input

    # Conv128-s
    # 5*5*128*128/2+128
    # ks*ks*nb_filters*channels/strides+bias(=nb_filters)

    # take 100 ms -> 4 frames.
    # if signal is 3 seconds, then take 100ms per 100ms and average out this network.
    # 8*8 = 64 features.
    inputs = Input(shape=[16, 16, 1])

    def conv_and_res_block(inp, filters, stage):
        o = Conv2D(filters,
                   kernel_size=5,
                   strides=2,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001), name='conv{}-s'.format(filters))(inp)
        o = clipped_relu(o)
        print(o)
        for i in range(3):
            o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
            print(o)
        return o

    x = conv_and_res_block(inputs, 64, stage=1)
    x = conv_and_res_block(x, 128, stage=2)
    x = conv_and_res_block(x, 256, stage=3)
    x = conv_and_res_block(x, 512, stage=4)

    x = Dense(512, name='affine')(x)
    x = Lambda(lambda y: y / K.max(y, axis=1), name='ln')(x)
    m = Model(inputs, x, name='convolutional')
    return m

# AveragePooling1D(name='average')(x)
# squeeze first! Reshape()

# test 1
# x = Lambda(lambda y: K.squeeze(y, axis=1))(x)
# x = AveragePooling1D(name='average')(x)

# test 2

# Maybe no averagePooling1D here but later on.
# def convolutional_model():
#     # http://cs231n.github.io/convolutional-networks/
#     # conv weights
#     # #params = ks * ks * nb_filters * num_channels_input
#
#     # Conv128-s
#     # 5*5*128*128/2+128
#     # ks*ks*nb_filters*channels/strides+bias(=nb_filters)
#
#     inputs = Input(shape=[20, 64, 4])
#
#     def conv_and_res_block(inp, filters, stage):
#         o = Conv2D(filters,
#                    kernel_size=5,
#                    strides=2,
#                    padding='same',
#                    kernel_initializer='glorot_uniform',
#                    kernel_regularizer=regularizers.l2(l=0.0001), name='conv{}-s'.format(filters))(inp)
#         o = clipped_relu(o)
#         print(o)
#         for i in range(3):
#             o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
#             print(o)
#         return o
#
#     x = conv_and_res_block(inputs, 64, stage=1)
#     x = conv_and_res_block(x, 128, stage=2)
#     x = conv_and_res_block(x, 256, stage=3)
#     x = conv_and_res_block(x, 512, stage=4)
#
#     # AveragePooling1D(name='average')(x)
#     # squeeze first! Reshape()
#
#     # test 1
#     # x = Lambda(lambda y: K.squeeze(y, axis=1))(x)
#     # x = AveragePooling1D(name='average')(x)
#
#     # test 2
#     x = Reshape((-1, 2048))(x)
#     x = AveragePooling1D(name='average')(x)
#
#     x = Dense(512, name='affine')(x)
#     x = Lambda(lambda y: y / K.max(y, axis=1), name='ln')(x)
#     m = Model(inputs, x, name='convolutional')
#     return m




#
# VERSION 1
# def convolutional_model():
#     # http://cs231n.github.io/convolutional-networks/
#     # conv weights
#     # #params = ks * ks * nb_filters * num_channels_input
#
#     inputs = Input(shape=[1, 20, 64])
#
#     def conv_and_res_block(inp, filters, stage):
#         o = Conv2D(filters,
#                    kernel_size=5,
#                    strides=2,
#                    padding='same',
#                    kernel_initializer='glorot_uniform',
#                    kernel_regularizer=regularizers.l2(l=0.0001))(inp)
#         o = clipped_relu(o)
#         print(o)
#         for i in range(3):
#             o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
#             print(o)
#         return o
#
#     x = conv_and_res_block(inputs, 64, stage=1)
#     x = conv_and_res_block(x, 128, stage=2)
#     x = conv_and_res_block(x, 256, stage=3)
#     x = conv_and_res_block(x, 512, stage=4)
#
#     # AveragePooling1D(name='average')(x)
#     # squeeze first! Reshape()
#
#     x = Lambda(lambda y: K.squeeze(y, axis=1))(x)
#     x = AveragePooling1D(name='average')(x)
#     x = Dense(512, name='affine')(x)
#     x = Lambda(lambda y: y / K.max(y, axis=1), name='ln')(x)
#     m = Model(inputs, x, name='convolutional')
#     return m
