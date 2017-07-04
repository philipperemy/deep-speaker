import sys
from time import time

import numpy as np

from constants import BATCH_SIZE
from librispeech_wav_reader import read_librispeech_structure
from models import convolutional_model
from next_batch import stochastic_mini_batch
from triplet_loss import deep_speaker_loss


def main(libri_dir):
    libri = read_librispeech_structure(libri_dir)
    batch = stochastic_mini_batch(libri, batch_size=BATCH_SIZE)
    x, y = batch.to_inputs()
    b = x[0]
    num_frames = b.shape[0]

    model = convolutional_model(batch_input_shape=[BATCH_SIZE * num_frames] + list(b.shape[1:]),
                                batch_size=BATCH_SIZE, num_frames=num_frames)
    model.compile(optimizer='adam', loss=deep_speaker_loss)

    print(model.summary())
    grad_steps = 0
    orig_time = time()
    while True:
        batch = stochastic_mini_batch(libri, batch_size=BATCH_SIZE)
        x, y = batch.to_inputs()

        # output.shape = (3, 383, 32, 32, 3) something like this
        # explanation  = (batch_size, num_frames, width, height, channels)
        x = np.reshape(x, (BATCH_SIZE * num_frames, b.shape[2], b.shape[2], b.shape[3]))

        # we don't need to use the targets y, because we know by the convention that:
        # we have [anchors, positive examples, negative examples]. The loss only uses x and
        # can determine if a sample is an anchor, positive or negative sample.
        stub_targets = np.random.uniform(size=(x.shape[0], 1))
        loss = model.train_on_batch(x, stub_targets)
        print('batch #{0} processed in {1:.2f}s, training loss = {2}.'.format(grad_steps, time() - orig_time, loss))
        grad_steps += 1
        orig_time = time()


if __name__ == '__main__':
    arguments = sys.argv
    assert len(arguments) == 2, 'Usage: python3 {} <libri_speech_wav_folder>'.format(arguments[0])
    main(arguments[1])
