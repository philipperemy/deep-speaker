import os
from collections import deque
from glob import glob

import numpy as np
import tensorflow.keras.backend as K
from natsort import natsorted
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Lambda, Flatten
from tensorflow.keras.optimizers import Adam

from batcher import KerasConverter, TripletBatcher
from constants import BATCH_SIZE, CHECKPOINTS_SOFTMAX_DIR, CHECKPOINTS_TRIPLET_DIR, NUM_FRAMES, NUM_FBANKS
from triplet_loss import deep_speaker_loss


# - Triplet Loss for embeddings
# - Softmax for pre-training
def triplet_softmax_model(batch_input_shape,
                          num_speakers_softmax,
                          emb_trainable=True,
                          normalize_embeddings=False):
    inp = Input(batch_shape=batch_input_shape)
    x = Conv2D(filters=64, kernel_size=5, strides=1, activation='relu')(inp)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Conv2D(filters=128, kernel_size=5, strides=1, activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Flatten()(x)
    embeddings = Dense(128, activation='sigmoid', name='fc1', trainable=emb_trainable)(x)
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
    # checkpoint = ModelCheckpoint(monitor='val_acc', filepath='checkpoints/model_{grad_step:02d}_{val_acc:.3f}.h5',
    #                              save_best_only=True)
    # if the accuracy does not increase by 1.0% over 10 epochs, we stop the training.
    # early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=100, verbose=1, mode='max')
    #
    # if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    # reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=1)

    # anchor and positive = first one.
    # negative = second one.
    # order is [anchor, positive, negative].

    triplet_batcher = TripletBatcher(kx_train, ky_train, kx_test, ky_test)

    test_every = 10
    train_deque_size = 200
    train_overall_loss_emb = deque(maxlen=train_deque_size)
    test_overall_loss_emb = deque(maxlen=train_deque_size // test_every)
    loss_file = open(os.path.join(CHECKPOINTS_TRIPLET_DIR, 'losses.txt'), 'w')

    for grad_step in range(int(1e12)):

        if grad_step % test_every == 0:  # test step.
            batch_x, batch_y = triplet_batcher.get_batch(batch_size, is_test=True)
            test_loss_values = m.test_on_batch(x=batch_x, y=batch_y)
            test_loss = dict(zip(m.metrics_names, test_loss_values))
            test_overall_loss_emb.append(test_loss['embeddings_loss'])

        # train step.
        batch_x, batch_y = triplet_batcher.get_batch(batch_size, is_test=False)
        train_loss_values = m.train_on_batch(x=batch_x, y=batch_y)
        train_loss = dict(zip(m.metrics_names, train_loss_values))
        train_overall_loss_emb.append(train_loss['embeddings_loss'])

        if grad_step % 100 == 0:
            format_str = 'step: {0:,}, train_loss: {1:.4f}, test_loss: {2:.4f}.'
            tr_mean_loss = np.mean(train_overall_loss_emb)
            te_mean_loss = np.mean(test_overall_loss_emb)
            print(format_str.format(grad_step, tr_mean_loss, te_mean_loss))
            loss_file.write(','.join([str(grad_step), f'{tr_mean_loss:.4f}', f'{te_mean_loss:.4f}']) + '\n')
            loss_file.flush()

        if grad_step % 10_000 == 0:
            print('Saving...')
            checkpoint_file = os.path.join(CHECKPOINTS_TRIPLET_DIR, f'triplet_model_{grad_step}.h5')
            m.save_weights(checkpoint_file, overwrite=True)

    loss_file.close()


def fit_model_softmax(m, kx_train, ky_train, kx_test, ky_test, batch_size=BATCH_SIZE, max_epochs=1000, initial_epoch=0):
    checkpoint = ModelCheckpoint(filepath=CHECKPOINTS_SOFTMAX_DIR + '/unified_model_checkpoints_{epoch}.h5', period=10)
    # if the accuracy does not increase by 1.0% over 10 epochs, we stop the training.
    # 100 was the value before.
    early_stopping = EarlyStopping(monitor='val_softmax_accuracy', min_delta=0.01, patience=100, verbose=1, mode='max')

    # if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    reduce_lr = ReduceLROnPlateau(monitor='val_softmax_accuracy', factor=0.5, patience=10, min_lr=0.0001, verbose=1)

    max_len_train = len(kx_train) - len(kx_train) % batch_size
    kx_train = kx_train[0:max_len_train]
    ky_train = ky_train[0:max_len_train]
    max_len_test = len(kx_test) - len(kx_test) % batch_size
    kx_test = kx_test[0:max_len_test]
    ky_test = ky_test[0:max_len_test]

    print('The embedding loss here does not make sense. Do not get fooled by it. Triplets are not present here.')
    print('We train the embedding weights first.')

    # class WarningCallback(Callback):
    #     def on_epoch_end(self, epoch, logs=None):
    #         print('The embedding loss here does not make sense. Do not get fooled by it. '
    #               'Triplets are not generated here. We train the embedding weights first.')

    m.fit(x=kx_train,
          y={'embeddings': ky_train, 'softmax': ky_train},
          batch_size=batch_size,
          epochs=initial_epoch + max_epochs,
          initial_epoch=initial_epoch,
          verbose=1,
          validation_data=(kx_test, {'embeddings': ky_test, 'softmax': ky_test}),
          callbacks=[early_stopping, reduce_lr, checkpoint])


def start_training(kc: KerasConverter, loss_on_softmax, loss_on_embeddings, normalize_embeddings):
    # loss_on_softmax = True
    # loss_on_embeddings = True
    freeze_embedding_weights = False
    # normalize_embeddings = True
    if not os.path.exists(CHECKPOINTS_SOFTMAX_DIR):
        os.makedirs(CHECKPOINTS_SOFTMAX_DIR)
    if not os.path.exists(CHECKPOINTS_TRIPLET_DIR):
        os.makedirs(CHECKPOINTS_TRIPLET_DIR)

    if not loss_on_softmax and not loss_on_embeddings:
        print('Please provide at least --loss_on_softmax or --loss_on_embeddings.')
        exit(1)

    emb_trainable = True
    if freeze_embedding_weights:
        print('FrEeZiNg tHe eMbeDdInG wEiGhTs.')
        emb_trainable = False

    m = triplet_softmax_model(
        batch_input_shape=[BATCH_SIZE, NUM_FRAMES, NUM_FBANKS, 3],
        num_speakers_softmax=len(kc.categorical_speakers.speaker_ids),
        emb_trainable=emb_trainable,
        normalize_embeddings=normalize_embeddings
    )

    checkpoints = natsorted(glob(os.path.join(CHECKPOINTS_SOFTMAX_DIR, '*.h5')))

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