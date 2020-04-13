import unittest

import numpy as np

from batcher import KerasConverter, TripletBatcherSelectHardNegatives, TripletBatcher
from constants import NUM_FBANKS, NUM_FRAMES, CHECKPOINTS_TRIPLET_DIR, CHECKPOINTS_SOFTMAX_DIR
from conv_models import DeepSpeakerModel
from triplet_loss import deep_speaker_loss
from utils import load_best_checkpoint


class BatcherTest(unittest.TestCase):

    def test_super_loss(self):
        working_dir = '/media/philippe/8TB/deep-speaker'
        # by construction this super loss should be much higher than the normal loss.
        # we select batches this way.
        batch_input_shape = [None, NUM_FRAMES, NUM_FBANKS, 1]
        print('Testing with the triplet loss.')
        dsm = DeepSpeakerModel(batch_input_shape, include_softmax=False)
        triplet_checkpoint = load_best_checkpoint(CHECKPOINTS_TRIPLET_DIR)
        pre_training_checkpoint = load_best_checkpoint(CHECKPOINTS_SOFTMAX_DIR)
        if triplet_checkpoint is not None:
            print(f'Loading triplet checkpoint: {triplet_checkpoint}.')
            dsm.m.load_weights(triplet_checkpoint)
        elif pre_training_checkpoint is not None:
            print(f'Loading pre-training checkpoint: {pre_training_checkpoint}.')
            # If `by_name` is True, weights are loaded into layers only if they share the
            # same name. This is useful for fine-tuning or transfer-learning models where
            # some of the layers have changed.
            dsm.m.load_weights(pre_training_checkpoint, by_name=True)
        dsm.m.compile(optimizer='adam', loss=deep_speaker_loss)
        kc = KerasConverter(working_dir)
        super_batcher = TripletBatcherSelectHardNegatives(kc.kx_train, kc.ky_train, kc.kx_test, kc.ky_test, dsm)
        simple_batcher = TripletBatcher(kc.kx_train, kc.ky_train, kc.kx_test, kc.ky_test)
        batch_size = 3
        simple_loss = []
        super_loss = []
        while True:
            super_bx, super_by = super_batcher.get_batch(batch_size, is_test=False)
            simple_bx, simple_by = simple_batcher.get_batch(batch_size, is_test=False)
            super_loss.append(dsm.m.evaluate(super_bx, super_by))
            simple_loss.append(dsm.m.evaluate(simple_bx, simple_by))
            print(np.mean(simple_loss), np.mean(super_loss))


if __name__ == '__main__':
    unittest.main()
