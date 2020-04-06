import os
import pickle
from collections import deque
from glob import glob

import numpy as np
from natsort import natsorted
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from batcher import KerasConverter, TripletBatcher
from constants import BATCH_SIZE, CHECKPOINTS_SOFTMAX_DIR, CHECKPOINTS_TRIPLET_DIR, NUM_FRAMES, NUM_FBANKS, \
    PRE_TRAINING_WEIGHTS_FILE
from conv_models import DeepSpeakerModel
from triplet_loss import deep_speaker_loss


def fit_model(dsm: DeepSpeakerModel, kx_train, ky_train, kx_test, ky_test, batch_size=BATCH_SIZE):
    batcher = TripletBatcher(kx_train, ky_train, kx_test, ky_test)

    test_every = 10
    train_deque_size = 200
    train_overall_loss_emb = deque(maxlen=train_deque_size)
    test_overall_loss_emb = deque(maxlen=train_deque_size // test_every)
    loss_file = open(os.path.join(CHECKPOINTS_TRIPLET_DIR, 'losses.txt'), 'w')

    for grad_step in range(int(1e12)):

        if grad_step % test_every == 0:  # test step.
            batch_x, batch_y = batcher.get_batch(batch_size, is_test=True)
            test_loss_values = dsm.m.test_on_batch(x=batch_x, y=batch_y)
            test_loss = dict(zip(dsm.m.metrics_names, test_loss_values))
            test_overall_loss_emb.append(test_loss['embeddings_loss'])

        # train step.
        batch_x, batch_y = batcher.get_batch(batch_size, is_test=False)
        train_loss_values = dsm.m.train_on_batch(x=batch_x, y=batch_y)
        train_loss = dict(zip(dsm.m.metrics_names, train_loss_values))
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
            dsm.m.save_weights(checkpoint_file, overwrite=True)

    loss_file.close()


def fit_model_softmax(dsm: DeepSpeakerModel, kx_train, ky_train, kx_test, ky_test,
                      batch_size=BATCH_SIZE, max_epochs=1000, initial_epoch=0):
    triplet_checkpoint_filename = PRE_TRAINING_WEIGHTS_FILE

    class ModelTripletCheckpoint(Callback):

        def on_epoch_end(self, epoch, logs=None):
            weights = dsm.get_weights()
            with open(triplet_checkpoint_filename, 'wb') as w:
                pickle.dump(weights, w)

    checkpoint = ModelCheckpoint(filepath=CHECKPOINTS_SOFTMAX_DIR + '/unified_model_checkpoints_{epoch}.h5', period=10)
    # if the accuracy does not increase by 1.0% over 10 epochs, we stop the training.
    # 100 was the value before.
    early_stopping = EarlyStopping(monitor='val_softmax_accuracy', min_delta=0.01, patience=100, verbose=1, mode='max')

    # if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    reduce_lr = ReduceLROnPlateau(monitor='val_softmax_accuracy', factor=0.5, patience=10, min_lr=0.0001, verbose=1)

    triplet_checkpoint = ModelTripletCheckpoint()

    max_len_train = len(kx_train) - len(kx_train) % batch_size
    kx_train = kx_train[0:max_len_train]
    ky_train = ky_train[0:max_len_train]
    max_len_test = len(kx_test) - len(kx_test) % batch_size
    kx_test = kx_test[0:max_len_test]
    ky_test = ky_test[0:max_len_test]

    dsm.m.fit(x=kx_train,
              y=ky_train,
              batch_size=batch_size,
              epochs=initial_epoch + max_epochs,
              initial_epoch=initial_epoch,
              verbose=1,
              validation_data=(kx_test, ky_test),
              callbacks=[early_stopping, reduce_lr, checkpoint, triplet_checkpoint])


def start_training(kc: KerasConverter, pre_training_phase=True):
    if not os.path.exists(CHECKPOINTS_SOFTMAX_DIR):
        os.makedirs(CHECKPOINTS_SOFTMAX_DIR)
    if not os.path.exists(CHECKPOINTS_TRIPLET_DIR):
        os.makedirs(CHECKPOINTS_TRIPLET_DIR)

    batch_input_shape = [None, NUM_FRAMES, NUM_FBANKS, 3]

    if pre_training_phase:
        print('Softmax pre-training.')
        num_speakers_softmax = len(kc.categorical_speakers.speaker_ids)
        dsm = DeepSpeakerModel(batch_input_shape, include_softmax=True, num_speakers_softmax=num_speakers_softmax)
        dsm.m.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        checkpoints = natsorted(glob(os.path.join(CHECKPOINTS_SOFTMAX_DIR, '*.h5')))
        initial_epoch = 0
        if len(checkpoints) != 0:
            checkpoint_file = checkpoints[-1]
            initial_epoch = int(checkpoint_file.split('/')[-1].split('.')[0].split('_')[-1])
            print(f'Initial epoch is {initial_epoch}.')
            print(f'Loading softmax checkpoint: {checkpoint_file}.')
            dsm.m.load_weights(checkpoint_file)  # latest one.
        fit_model_softmax(dsm, kc.kx_train, kc.ky_train, kc.kx_test, kc.ky_test, initial_epoch=initial_epoch)
    else:
        print('Training with the triplet loss.')
        dsm = DeepSpeakerModel(batch_input_shape, include_softmax=False)
        dsm.m.compile(optimizer=Adam(lr=0.01), loss=deep_speaker_loss)
        weights_file = PRE_TRAINING_WEIGHTS_FILE
        checkpoints = natsorted(glob(os.path.join(CHECKPOINTS_TRIPLET_DIR, '*.h5')))
        if len(checkpoints) != 0:
            checkpoint_file = checkpoints[-1]
            print(f'Loading triplet checkpoint: {checkpoint_file}.')
            dsm.m.load_weights(checkpoint_file)
        elif os.path.isfile(weights_file):
            print(f'Loading pre-training weights from: {weights_file}.')
            with open(weights_file) as r:
                w = pickle.load(r)
            dsm.m.set_weights(w)
        fit_model(dsm, kc.kx_train, kc.ky_train, kc.kx_test, kc.ky_test)
