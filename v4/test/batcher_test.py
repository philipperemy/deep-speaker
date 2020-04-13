import unittest

import numpy as np

from batcher import KerasConverter, TripletBatcherSelectHardNegatives, TripletBatcher
from constants import NUM_FBANKS, NUM_FRAMES
from conv_models import DeepSpeakerModel
from triplet_loss import deep_speaker_loss


class BatcherTest(unittest.TestCase):

    def test_super_loss(self):
        # by construction this super loss should be much higher than the normal loss.
        # we select batches this way.
        batch_input_shape = [None, NUM_FRAMES, NUM_FBANKS, 1]
        kc = KerasConverter('/media/philippe/8TB/deep-speaker')
        dsm = DeepSpeakerModel(batch_input_shape, include_softmax=False)
        dsm.m.compile(optimizer='adam', loss=deep_speaker_loss)
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
