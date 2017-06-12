import numpy as np
from python_speech_features import fbank

from models import convolutional_model

if __name__ == '__main__':
    signal = np.random.uniform(size=200)
    fbank(signal, samplerate=4000, nfilt=64)

    model = convolutional_model()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
