import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


def build_model_softmax(m):
    m.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.01),
              metrics=['accuracy'])


def fit_model_softmax(m, kx_train, ky_train, kx_test, ky_test, batch_size=30, max_epochs=1000):
    # checkpoint = ModelCheckpoint(monitor='val_acc', filepath='checkpoints/model_{epoch:02d}_{val_acc:.3f}.h5',
    #                              save_best_only=True)
    # if the accuracy does not increase by 1.0% over 10 epochs, we stop the training.
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=100, verbose=1, mode='max')

    # if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=1)

    max_len_train = len(kx_train) - len(kx_train) % batch_size

    kx_train = kx_train[0:max_len_train]
    ky_train = ky_train[0:max_len_train]

    max_len_test = len(kx_test) - len(kx_test) % batch_size
    kx_test = kx_test[0:max_len_test]
    ky_test = ky_test[0:max_len_test]

    m.fit(kx_train,
          ky_train,
          batch_size=batch_size,
          epochs=max_epochs,
          verbose=1,
          validation_data=(kx_test, ky_test),
          callbacks=[early_stopping, reduce_lr])


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
def get_model(num_classes):
    m = Sequential()
    m.add(Dense(200, batch_input_shape=[None, 39 * 10], activation='sigmoid'))
    m.add(Dense(num_classes, activation='softmax'))
    print(m.summary())
    return m
