import unittest

import numpy as np

from deep_speaker.triplet_loss import deep_speaker_loss


def opposite_positive_equal_negative_batch():
    # should be the highest
    b = np.random.uniform(low=-1, high=1, size=(18, 512))
    b[0] = -b[6]
    b[1] = -b[7]
    b[2] = -b[8]
    b[3] = -b[9]
    b[4] = -b[10]
    b[5] = -b[11]
    b[12] = b[0]
    b[13] = b[1]
    b[14] = b[2]
    b[15] = b[3]
    b[16] = b[4]
    b[17] = b[5]
    return b


def random_positive_random_negative_batch():
    # should be high
    b = np.random.uniform(low=-1, high=1, size=(18, 512))
    return b


def equal_positive_random_negative_batch():
    # should be low
    b = np.random.uniform(low=-1, high=1, size=(12, 512))
    b[0] = b[6]
    b[1] = b[7]
    b[2] = b[8]
    b[3] = b[9]
    b[4] = b[10]
    b[5] = b[11]
    return b


def equal_positive_opposite_negative_batch():
    # should be the lowest
    b = np.random.uniform(low=-1, high=1, size=(18, 512))
    b[0] = b[6]
    b[1] = b[7]
    b[2] = b[8]
    b[3] = b[9]
    b[4] = b[10]
    b[5] = b[11]
    b[12] = -b[0]
    b[13] = -b[1]
    b[14] = -b[2]
    b[15] = -b[3]
    b[16] = -b[4]
    b[17] = -b[5]
    return b


class TripleLossTest(unittest.TestCase):

    def test_1(self):
        # ANCHOR 1 (512,), index = 0
        # ANCHOR 2 (512,), index = 1
        # ANCHOR 3 (512,), index = 2
        # ANCHOR 4 (512,), index = 3
        # ANCHOR 5 (512,), index = 4
        # ANCHOR 6 (512,), index = 5
        # POS EX 1 (512,), index = 6
        # POS EX 2 (512,), index = 7
        # POS EX 3 (512,), index = 8
        # POS EX 4 (512,), index = 9
        # POS EX 5 (512,), index = 10
        # POS EX 6 (512,), index = 11
        # NEG EX 1 (512,), index = 12
        # NEG EX 2 (512,), index = 13
        # NEG EX 3 (512,), index = 14
        # NEG EX 4 (512,), index = 15
        # NEG EX 5 (512,), index = 16
        # NEG EX 6 (512,), index = 17

        x2 = 0  # nothing because not used.

        np.random.seed(123)
        highest_loss = deep_speaker_loss(x2, opposite_positive_equal_negative_batch()).numpy()
        high_loss = deep_speaker_loss(x2, random_positive_random_negative_batch())
        low_loss = deep_speaker_loss(x2, equal_positive_random_negative_batch())
        lowest_loss = deep_speaker_loss(x2, equal_positive_opposite_negative_batch())
        self.assertTrue(highest_loss >= high_loss >= low_loss >= lowest_loss)
