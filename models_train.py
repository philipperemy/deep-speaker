from time import time

import numpy as np

from models import convolutional_model
from pre_process import next_batch
from triplet_loss import deep_speaker_loss

if __name__ == '__main__':
    b = next_batch()
    batch_size = 3
    num_frames = b.shape[0]

    model = convolutional_model(batch_input_shape=[None] + list(b.shape[1:]))
    model.compile(optimizer='adam',
                  loss=deep_speaker_loss)

    print(model.summary())
    grad_steps = 0
    orig_time = time()
    while True:
        anc = next_batch()
        pos = next_batch()
        neg = next_batch()
        batch = np.concatenate([anc, pos, neg], axis=0)

        # this line should not raise an error
        # output.shape = (3, 383, 32, 32, 3)
        # explanation  = (batch_size, num_frames, width, height, channels)
        np.reshape(batch, (batch_size, num_frames, b.shape[2], b.shape[2], b.shape[3]))

        stub_targets = np.random.uniform(size=(batch.shape[0], 1))
        loss = model.train_on_batch(batch, stub_targets)[0]
        print('batch #{0} processed in {1:.2f}s, training loss = {2}.'.format(grad_steps, time() - orig_time, loss))
        grad_steps += 1
        orig_time = time()
