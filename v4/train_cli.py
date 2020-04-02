import os
from collections import deque
from glob import glob

import keras.backend as K
import numpy as np
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from keras.layers import Dense, Lambda
from keras.optimizers import Adam
from natsort import natsorted

from batcher import KerasConverter
from triplet_loss import deep_speaker_loss

BATCH_SIZE = 900


# - Triplet Loss for embeddings
# - Softmax for pre-training
def triplet_softmax_model(num_speakers_softmax, batch_size=BATCH_SIZE,
                          emb_trainable=True, normalize_embeddings=False):
    inp = Input(batch_shape=[batch_size, 39 * 10])
    embeddings = Dense(200, activation='sigmoid', name='fc1', trainable=emb_trainable)(inp)
    if normalize_embeddings:
        print('Embeddings will be normalized.')
        embeddings = Lambda(lambda y: K.l2_normalize(y, axis=1), name='normalization')(embeddings)
    embeddings = Lambda(lambda y: y, name='embeddings')(embeddings)  # just a trick to name a layer after if-else.
    softmax = Dense(num_speakers_softmax, activation='softmax', name='softmax')(embeddings)
    return Model(inputs=[inp], outputs=[embeddings, softmax])


def compile_triplet_softmax_model(m: Model, loss_on_softmax=True, loss_on_embeddings=False):
    losses = {
        'embeddings': deep_speaker_loss,
        'softmax': 'categorical_crossentropy',
    }

    loss_weights = {
        'embeddings': int(loss_on_embeddings),
        'softmax': int(loss_on_softmax),
    }

    print(losses)
    print(loss_weights)
    m.compile(optimizer=Adam(lr=0.0001),
              loss=losses,
              loss_weights=loss_weights,
              metrics=['accuracy'])


def fit_model(m, kx_train, ky_train, kx_test, ky_test,
              batch_size=BATCH_SIZE, max_grad_steps=1000000, initial_epoch=0):
    # TODO: use this callback checkpoint.
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

    def select_inputs_and_outputs_for_speaker(x, y, speaker_id_):
        indices = np.random.choice(np.where(y.argmax(axis=1) == speaker_id_)[0], size=batch_size // 3)
        return x[indices], y[indices]

    print()
    print()
    assert sorted(set(ky_test.argmax(axis=1))) == sorted(set(ky_train.argmax(axis=1)))
    num_different_speakers = len(set(ky_train.argmax(axis=1)))
    print('num different speakers =', num_different_speakers)
    deque_size = 100
    train_overall_loss_emb = deque(maxlen=deque_size)
    test_overall_loss_emb = deque(maxlen=deque_size)
    train_overall_loss_softmax = deque(maxlen=deque_size)
    test_overall_loss_softmax = deque(maxlen=deque_size)
    # TODO: not very much epoch here.
    for epoch in range(initial_epoch, max_grad_steps):
        two_different_speakers = np.random.choice(range(num_different_speakers), size=2, replace=False)
        anchor_positive_speaker = two_different_speakers[0]
        # negative_speaker = two_different_speakers[0]
        negative_speaker = two_different_speakers[1]
        assert negative_speaker != anchor_positive_speaker

        train_inputs_outputs = [
            select_inputs_and_outputs_for_speaker(kx_train, ky_train, anchor_positive_speaker),
            select_inputs_and_outputs_for_speaker(kx_train, ky_train, anchor_positive_speaker),
            select_inputs_and_outputs_for_speaker(kx_train, ky_train, negative_speaker)
        ]
        inputs = np.vstack([v[0] for v in train_inputs_outputs])
        outputs = np.vstack([v[1] for v in train_inputs_outputs])

        train_loss = m.train_on_batch(inputs, {'embeddings': outputs * 0, 'softmax': outputs})
        train_loss = dict(zip(m.metrics_names, train_loss))
        train_overall_loss_emb.append(train_loss['embeddings_loss'])
        train_overall_loss_softmax.append(train_loss['softmax_loss'])

        test_inputs_outputs = [
            select_inputs_and_outputs_for_speaker(kx_test, ky_test, anchor_positive_speaker),
            select_inputs_and_outputs_for_speaker(kx_test, ky_test, anchor_positive_speaker),
            select_inputs_and_outputs_for_speaker(kx_test, ky_test, negative_speaker)
        ]

        test_inputs = np.vstack([v[0] for v in test_inputs_outputs])
        test_outputs = np.vstack([v[1] for v in test_inputs_outputs])

        test_loss = m.test_on_batch(test_inputs, {'embeddings': test_outputs * 0, 'softmax': test_outputs})
        test_loss = dict(zip(m.metrics_names, test_loss))
        test_overall_loss_emb.append(test_loss['embeddings_loss'])
        test_overall_loss_softmax.append(test_loss['softmax_loss'])

        if epoch % 10 == 0:
            format_str = '{0}, train(emb, last {3}) = {1:.5f} test(emb, last {3}) = {2:.5f}.'
            print(format_str.format(str(epoch).zfill(6),
                                    np.mean(train_overall_loss_emb),
                                    np.mean(test_overall_loss_emb),
                                    deque_size))

        if epoch % 100 == 0:
            print('train metrics =', train_loss)
            print('test metrics =', test_loss)
            m.save_weights('checkpoints/unified_model_checkpoints_{}.h5'.format(epoch), overwrite=True)
            print('Last two speakers were {} and {}.'.format(anchor_positive_speaker, negative_speaker))
            print('Saving...')


def fit_model_softmax(m, kx_train, ky_train, kx_test, ky_test, batch_size=BATCH_SIZE, max_epochs=1000, initial_epoch=0):
    checkpoint = ModelCheckpoint(filepath='checkpoints/unified_model_checkpoints_{epoch}.h5',
                                 period=10)
    # if the accuracy does not increase by 1.0% over 10 epochs, we stop the training.
    early_stopping = EarlyStopping(monitor='val_softmax_acc', min_delta=0.01, patience=100, verbose=1, mode='max')

    # if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    reduce_lr = ReduceLROnPlateau(monitor='val_softmax_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=1)

    max_len_train = len(kx_train) - len(kx_train) % batch_size

    kx_train = kx_train[0:max_len_train]
    ky_train = ky_train[0:max_len_train]

    max_len_test = len(kx_test) - len(kx_test) % batch_size
    kx_test = kx_test[0:max_len_test]
    ky_test = ky_test[0:max_len_test]

    print('The embedding loss here does not make sense. Do not get fooled by it. Triplets are not present here.')
    print('We train the embedding weights first.')

    class WarningCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('The embedding loss here does not make sense. Do not get fooled by it. '
                  'Triplets are not generated here. We train the embedding weights first.')

    m.fit(kx_train,
          {'embeddings': ky_train, 'softmax': ky_train},
          batch_size=batch_size,
          epochs=initial_epoch + max_epochs,
          initial_epoch=initial_epoch,
          verbose=1,
          validation_data=(kx_test, {'embeddings': ky_test, 'softmax': ky_test}),
          callbacks=[early_stopping, reduce_lr, checkpoint, WarningCallback()])


def start_training(checkpoints_dir: str, kc: KerasConverter):
    loss_on_softmax = True
    loss_on_embeddings = True
    freeze_embedding_weights = False
    normalize_embeddings = True
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if not loss_on_softmax and not loss_on_embeddings:
        print('Please provide at least --loss_on_softmax or --loss_on_embeddings.')
        exit(1)

    emb_trainable = True
    if freeze_embedding_weights:
        print('FrEeZiNg tHe eMbeDdInG wEiGhTs.')
        emb_trainable = False

    m = triplet_softmax_model(num_speakers_softmax=len(kc.categorical_speakers.speaker_ids),
                              emb_trainable=emb_trainable,
                              normalize_embeddings=normalize_embeddings)

    checkpoints = natsorted(glob(os.path.join(checkpoints_dir, '*.h5')))

    compile_triplet_softmax_model(m, loss_on_softmax, loss_on_embeddings)
    print(m.summary())

    initial_epoch = 0
    if len(checkpoints) != 0:
        checkpoint_file = checkpoints[-1]
        initial_epoch = int(checkpoint_file.split('/')[-1].split('.')[0].split('_')[-1])
        print('Initial epoch is {}.'.format(initial_epoch))
        print('Loading checkpoint: {}.'.format(checkpoint_file))
        m.load_weights(checkpoint_file)  # latest one.

    if loss_on_softmax:
        print('Softmax pre-training.')
        fit_model_softmax(m, kc.kx_train, kc.ky_train, kc.kx_test, kc.ky_test, initial_epoch=initial_epoch)
    else:
        fit_model(m, kc.kx_train, kc.ky_train, kc.kx_test, kc.ky_test, initial_epoch=initial_epoch)
