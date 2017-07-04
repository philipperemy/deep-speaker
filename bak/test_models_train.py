import numpy as np

from constants import BATCH_NUM_TRIPLETS, NUM_FRAMES
from models import convolutional_model


def normalize_frames(m):
    return [(v - np.mean(v)) / np.std(v) for v in m]


if __name__ == '__main__':
    network_inputs = np.random.uniform(size=(BATCH_NUM_TRIPLETS, NUM_FRAMES, 16, 16, 1))

    model = convolutional_model(batch_input_shape=(BATCH_NUM_TRIPLETS * NUM_FRAMES, 16, 16, 1))

    from triplet_loss import deep_speaker_loss
    model.compile(optimizer='adam',
                  loss=deep_speaker_loss,
                  metrics=['accuracy'])

    network_inputs = np.reshape(network_inputs, (-1, 16, 16, 1))

    output = model.predict(network_inputs)

    # stub_targets = np.expand_dims([0] * BATCH_SIZE * NUM_FRAMES, axis=1)
    stub_targets = np.random.uniform(size=(BATCH_NUM_TRIPLETS * NUM_FRAMES, 512))
    print(model.train_on_batch(network_inputs, stub_targets))

    # from triplet_loss import deep_speaker_loss

    # deep_speaker_loss(output, None)

    # print(model.predict(np.expand_dims(network_inputs[0][0], axis=0)).shape)
    # print(model.predict(np.reshape(network_inputs, (-1, 16, 16, 1))).shape)
