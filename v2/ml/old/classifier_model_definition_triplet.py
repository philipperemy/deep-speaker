from collections import deque
from glob import glob

import numpy as np
from keras import Input, Model
from keras.layers import Dense
from keras.models import load_model
from natsort import natsorted

from ml.triplet_loss import deep_speaker_loss


def build_model(m):
    m.compile(loss=deep_speaker_loss,
              optimizer='adam',
              metrics=['accuracy'])


def fit_model(m, kx_train, ky_train, kx_test, ky_test, batch_size, max_epochs=100000):
    # checkpoint = ModelCheckpoint(monitor='val_acc', filepath='checkpoints/model_{epoch:02d}_{val_acc:.3f}.h5',
    #                              save_best_only=True)
    # if the accuracy does not increase by 1.0% over 10 epochs, we stop the training.
    # early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=100, verbose=1, mode='max')
    #
    # if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    # reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=1)

    # anchor and positive = first one.
    # negative = second one.
    # order is [anchor, positive, negative].

    def select_inputs_for_speaker(x, y, speaker_id_):
        return x[np.random.choice(np.where(y.argmax(axis=1) == speaker_id_)[0], size=batch_size // 3)]

    print()
    print()
    te_sp = sorted(set(ky_test.argmax(axis=1)))
    tr_sp = sorted(set(ky_train.argmax(axis=1)))
    assert len(te_sp) == len(tr_sp)
    assert te_sp == tr_sp
    num_different_speakers = len(set(ky_train.argmax(axis=1)))
    print('num different speakers =', num_different_speakers)
    train_overall_loss = deque(maxlen=1000)
    test_overall_loss = deque(maxlen=1000)
    for epoch in range(max_epochs):
        two_different_speakers = np.random.choice(range(num_different_speakers), size=2, replace=False)
        anchor_positive_speaker = two_different_speakers[0]
        # negative_speaker = two_different_speakers[0]
        negative_speaker = two_different_speakers[1]

        inputs = np.vstack([
            select_inputs_for_speaker(kx_train, ky_train, anchor_positive_speaker),
            select_inputs_for_speaker(kx_train, ky_train, anchor_positive_speaker),
            select_inputs_for_speaker(kx_train, ky_train, negative_speaker)
        ])

        loss = m.train_on_batch(inputs, ky_train[0:batch_size] * 0)
        train_overall_loss.append(loss)

        if epoch % 100 == 0:
            m.save_weights('checkpoints/checkpoints_{}.h5'.format(epoch), overwrite=True)
            print('Last two speakers were {} and {}'.format(anchor_positive_speaker, negative_speaker))
            print('Saving...')

        test_inputs = np.vstack([
            select_inputs_for_speaker(kx_test, ky_test, anchor_positive_speaker),
            select_inputs_for_speaker(kx_test, ky_test, anchor_positive_speaker),
            select_inputs_for_speaker(kx_test, ky_test, negative_speaker)
        ])
        test_loss = m.test_on_batch(test_inputs, ky_test[0:batch_size] * 0)
        test_overall_loss.append(test_loss)

        if epoch % 10 == 0:
            print(epoch, loss[0], np.mean(train_overall_loss), test_loss[0], np.mean(test_overall_loss))


def inference_model(m, input_list):
    log_probabilities = predict(m, input_list, log=True)
    k_star = np.argmax(np.sum(log_probabilities, axis=0))
    return k_star


def predict(m, norm_inputs, log=False):
    probabilities = m.predict(np.array(norm_inputs))
    if log:
        probabilities = np.log(probabilities + 0.000001)  # for stability.
    return probabilities


# No need to have regularization. The dataset is really big and the model really small.
def get_model(batch_size=3 * 10):
    inp = Input(batch_shape=[batch_size, 39 * 10])
    x = Dense(200, activation='sigmoid')(inp)
    m = Model(inputs=inp, outputs=x)
    # m = Sequential()
    # m.add(Dense(200, batch_input_shape=[None, 39 * 10], activation='sigmoid'))
    # m.add(Dense(num_classes, activation='softmax'))
    # print(m.summary())
    return m


def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    import keras.backend as K
    # print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
    return activations


if __name__ == '__main__':
    checkpoints = natsorted(glob('checkpoints/*.h5'))
    assert len(checkpoints) != 0, 'No checkpoints found.'
    checkpoint_file = checkpoints[-1]
    print('Loading [{}]'.format(checkpoint_file))
    m = load_model(checkpoint_file)
    print(m.predict(np.array([np.random.uniform(size=390)])).argmax())
    print(m.predict(np.array([np.random.normal(size=390)])).argmax())

    v1_m = np.array(
        [get_activations(m, np.array([np.random.normal(size=390)]), print_shape_only=True, layer_name=None)[0][0] for _
         in range(1000)])
    v2_m = np.array(
        [get_activations(m, np.array([np.random.uniform(size=390)]), print_shape_only=True, layer_name=None)[0][0] for
         _ in range(1000)])


    def mse(a, b):
        return np.mean(np.square(np.subtract(a, b)))


    from scipy.spatial.distance import cosine

    print(np.mean([cosine(u, v) for (u, v) in zip(v1_m[:-1], v1_m[1:])]))
    print(np.mean([cosine(u, v) for (u, v) in zip(v2_m, v1_m)]))

    print(mse(v1_m[:-1], v1_m[1:]))
    print(mse(v2_m, v1_m))
