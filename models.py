import keras.backend as K
from keras import layers
from keras import regularizers
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model

layers_dict = dict()


def get(obj):
    layer_name = obj.name
    if layer_name not in layers_dict:
        print('-> Creating layer [{}]'.format(layer_name))
        # create it
        layers_dict[layer_name] = obj
    else:
        print('-> Using layer [{}]'.format(layer_name))
    return layers_dict[layer_name]


# no need to share the weights here because it does not exist.
def clipped_relu(inputs):
    return get(Lambda(lambda y: K.minimum(K.maximum(y, 0), 20), name='clipped_relu'))(inputs)


def identity_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res{}_{}_branch'.format(stage, block)

    x = get(Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001),
                   name=conv_name_base + '_2a'))(input_tensor)
    x = get(BatchNormalization(name=conv_name_base + '_2a_bn'))(x)
    # x = Activation('relu')(x)
    x = clipped_relu(x)

    x = get(Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001),
                   name=conv_name_base + '_2b'))(x)
    x = get(BatchNormalization(name=conv_name_base + '_2b_bn'))(x)

    # 1x1 conv
    x = get(Conv2D(filters,
                   kernel_size=1,
                   strides=1,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001),
                   name=conv_name_base + '2c_bn'))(x)

    x = layers.add([x, input_tensor])

    x = get(BatchNormalization(name=conv_name_base + '2d_bn'))(x)
    # x = Activation('relu')(x)
    x = clipped_relu(x)
    return x


# def aggregator_model()

def convolutional_model(input_shapes=(16, 16, 1), num_frames=4):
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

    def conv_and_res_block(inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        o = get(Conv2D(filters,
                       kernel_size=5,
                       strides=2,
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=regularizers.l2(l=0.0001), name=conv_name))(inp)
        o = clipped_relu(o)
        for i in range(3):
            o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
        return o

    def cnn_component(inp):
        x_ = conv_and_res_block(inp, 64, stage=1)
        x_ = conv_and_res_block(x_, 128, stage=2)
        x_ = conv_and_res_block(x_, 256, stage=3)
        x_ = conv_and_res_block(x_, 512, stage=4)
        return x_

    inputs_list = []
    outputs_list = []
    for i in range(num_frames):
        inputs = Input(shape=input_shapes)
        inputs_list.append(inputs)
        x = cnn_component(inputs)
        outputs_list.append(x)

    def lambda_average(inp):
        out = inp[0]
        t = len(inp)
        for j in range(1, t):
            out += inp[j]
        out *= (1.0 / t)
        return out

    average_layer = get(Lambda(lambda y: lambda_average(y), name='average'))  # average
    x = average_layer(outputs_list)
    x = Dense(512, name='affine')(x)
    x = Lambda(lambda y: K.squeeze(K.squeeze(y, axis=1), axis=1))(x)
    x = Lambda(lambda y: y / K.max(y, axis=1), name='ln')(x)
    m = Model(inputs_list, x, name='convolutional')
    return m
