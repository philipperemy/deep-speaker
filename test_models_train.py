import numpy as np

from models import convolutional_model
from triplet_loss import deep_speaker_loss


def normalize_frames(m):
    return [(v - np.mean(v)) / np.std(v) for v in m]


if __name__ == '__main__':
    network_inputs = np.random.uniform(size=(3, 16, 16, 1))

    model = convolutional_model(input_shapes=list(network_inputs[0].shape),
                                num_frames=len(network_inputs))

    model.compile(optimizer='adam',
                  loss=deep_speaker_loss,
                  metrics=['accuracy'])

    inputs = list(np.expand_dims(network_inputs, axis=1))
    model.fit(inputs, np.expand_dims([0] * len(inputs), axis=1))

    print(model.summary())
