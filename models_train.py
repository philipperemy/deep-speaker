from time import time

import numpy as np

from constants import BATCH_NUM_TRIPLETS, DATASET_DIR, CHECKPOINT_FOLDER
from librispeech_wav_reader import read_librispeech_structure
from models import convolutional_model
from next_batch import stochastic_mini_batch
from triplet_loss import deep_speaker_loss
from utils import get_last_checkpoint_if_any, create_dir_and_delete_content


def main(libri_dir=DATASET_DIR):
    print('Looking for audio [wav] files in {}.'.format(libri_dir))
    libri = read_librispeech_structure(libri_dir)

    if len(libri) == 0:
        print('Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh')

    batch = stochastic_mini_batch(libri, batch_size=BATCH_NUM_TRIPLETS)
    batch_size = BATCH_NUM_TRIPLETS * 3  # A triplet has 3 parts.
    x, y = batch.to_inputs()
    b = x[0]
    num_frames = b.shape[0]
    print('num_frames = ', num_frames)

    model = convolutional_model(batch_input_shape=[batch_size * num_frames] + list(b.shape[1:]),
                                batch_size=batch_size, num_frames=num_frames)
    print(model.summary())

    print('Compiling the model...', end=' ')
    model.compile(optimizer='adam', loss=deep_speaker_loss)
    print('[DONE]')

    last_checkpoint = get_last_checkpoint_if_any(CHECKPOINT_FOLDER)
    if last_checkpoint is not None:
        print('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint), end=' ')
        model.load_weights(last_checkpoint)
        print('[DONE]')

    print('Starting training...')
    grad_steps = 0
    orig_time = time()
    while True:
        batch = stochastic_mini_batch(libri, batch_size=BATCH_NUM_TRIPLETS)
        x, _ = batch.to_inputs()

        # output.shape = (3, 383, 32, 32, 3) something like this
        # explanation  = (batch_size, num_frames, width, height, channels)
        x = np.reshape(x, (batch_size * num_frames, b.shape[2], b.shape[2], b.shape[3]))

        # we don't need to use the targets y, because we know by the convention that:
        # we have [anchors, positive examples, negative examples]. The loss only uses x and
        # can determine if a sample is an anchor, positive or negative sample.
        stub_targets = np.random.uniform(size=(x.shape[0], 1))
        # result = model.predict(x, batch_size=x.shape[0])
        # print(result.shape)
        # np.set_printoptions(precision=2)
        # print(result[0:20, 0:5])

        print('-' * 80)
        print('== Presenting batch #{0}'.format(grad_steps))
        print(batch.libri_batch)
        loss = model.train_on_batch(x, stub_targets)
        print('== Processed in {0:.2f}s by the network, training loss = {1}.'.format(time() - orig_time, loss))
        grad_steps += 1
        orig_time = time()

        # checkpoints are really heavy so let's just keep the last one.
        create_dir_and_delete_content(CHECKPOINT_FOLDER)
        model.save_weights('{0}/model_{1}_{2:.3f}.h5'.format(CHECKPOINT_FOLDER, grad_steps, loss))


if __name__ == '__main__':
    main()
