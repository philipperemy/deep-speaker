import logging
from time import time

import numpy as np
import sys

import ds_constants as c
from librispeech_wav_reader import read_librispeech_structure
from models import convolutional_model
from next_batch import stochastic_mini_batch
from triplet_loss import deep_speaker_loss
from utils import get_last_checkpoint_if_any, create_dir_and_delete_content


def main(libri_dir=c.DATASET_DIR):
    logging.info('Looking for audio [wav] files in {}.'.format(libri_dir))
    libri = read_librispeech_structure(libri_dir)

    if len(libri) == 0:
        logging.warning('Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh')
        exit(1)

    batch = stochastic_mini_batch(libri, batch_size=c.BATCH_NUM_TRIPLETS)
    batch_size = c.BATCH_NUM_TRIPLETS * 3  # A triplet has 3 parts.
    x, y = batch.to_inputs()
    b = x[0]
    num_frames = b.shape[0]
    logging.info('num_frames = {}'.format(num_frames))

    batch_shape = [batch_size * num_frames] + list(b.shape[1:])
    logging.info('batch shape: {}'.format(batch_shape))
    logging.info('batch size: {}'.format(batch_size))
    model = convolutional_model(batch_input_shape=batch_shape,
                                batch_size=batch_size, num_frames=num_frames)
    logging.info(model.summary())

    logging.info('Compiling the model...')
    model.compile(optimizer='adam', loss=deep_speaker_loss)
    logging.info('[DONE]')

    grad_steps = 0
    last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER)
    if last_checkpoint is not None:
        logging.info('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
        model.load_weights(last_checkpoint)
        grad_steps = int(last_checkpoint.split('_')[-2])
        logging.info('[DONE]')

    logging.info('Starting training...')
    orig_time = time()

    while True:
        grad_steps += 1
        batch = stochastic_mini_batch(libri, batch_size=c.BATCH_NUM_TRIPLETS)
        x, _ = batch.to_inputs()

        # output.shape = (3, 383, 32, 32, 3) something like this
        # explanation  = (batch_size, num_frames, width, height, channels)
        logging.info('x.shape before reshape: {}'.format(x.shape))
        x = np.reshape(x, (batch_size * num_frames, b.shape[2], b.shape[2], b.shape[3]))
        logging.info('x.shape after  reshape: {}'.format(x.shape))

        # we don't need to use the targets y, because we know by the convention that:
        # we have [anchors, positive examples, negative examples]. The loss only uses x and
        # can determine if a sample is an anchor, positive or negative sample.
        stub_targets = np.random.uniform(size=(x.shape[0], 1))
        # result = model.predict(x, batch_size=x.shape[0])
        # logging.info(result.shape)
        # np.set_printoptions(precision=2)
        # logging.info(result[0:20, 0:5])

        logging.info('-' * 80)
        logging.info('== Presenting batch #{0}'.format(grad_steps))
        logging.info(batch.libri_batch)
        loss = model.train_on_batch(x, stub_targets)
        logging.info('== Processed in {0:.2f}s by the network, training loss = {1}.'.format(time() - orig_time, loss))
        orig_time = time()

        # record training loss
        with open(c.LOSS_FILE, "a") as f:
            f.write("{0},{1}\n".format(grad_steps, loss))

        # checkpoints are really heavy so let's just keep the last one.
        create_dir_and_delete_content(c.CHECKPOINT_FOLDER)
        model.save_weights('{0}/model_{1}_{2:.5f}.h5'.format(c.CHECKPOINT_FOLDER, grad_steps, loss))



if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler(stream=sys.stdout)], level=logging.INFO,
                        format='%(asctime)-15s [%(levelname)s] %(filename)s/%(funcName)s | %(message)s')
    main()
