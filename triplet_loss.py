import keras.backend as K

from constants import *

alpha = 0.1


# K.sum(anchor * positive_ex, axis=1).eval() == K.batch_dot(anchor, positive_ex, axes=1)

def batch_norm(x):
    return K.sqrt(K.sum(K.square(x), axis=1))


def cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    return K.squeeze(K.batch_dot(x1, x2, axes=1), axis=1) / (batch_norm(x1) * batch_norm(x2))


def deep_speaker_loss(x1, x2):
    # x1.shape = (batch_size, embedding_size)
    # x2.shape = (batch_size, embedding_size)
    # CONVENTION: Input is:
    # concat(BATCH_SIZE * [ANCHOR, POSITIVE_EX, NEGATIVE_EX] * NUM_FRAMES)
    # EXAMPLE:
    # BATCH_NUM_TRIPLETS = 3, NUM_FRAMES = 2
    # _____________________________________________________
    # ANCHOR 1 (512,)
    # ANCHOR 2 (512,)
    # ANCHOR 3 (512,)
    # POS EX 1 (512,)
    # POS EX 2 (512,)
    # POS EX 3 (512,)
    # NEG EX 1 (512,)
    # NEG EX 2 (512,)
    # NEG EX 3 (512,)
    # ANCHOR 1 # FROM HERE (INCLUDED), THIS IS THE SAME BLOCK AS ABOVE.
    # ANCHOR 2 # WE ADD IT BECAUSE WE WANT TO MATCH THE SIZE FOR KERAS.
    # ANCHOR 3 # BATCH_SIZE * NUM_FRAMES => BATCH_SIZE => BATCH_SIZE * NUM_FRAMES
    # POS EX 1 (512,)
    # POS EX 2 (512,)
    # POS EX 3 (512,)
    # NEG EX 1 (512,)
    # NEG EX 2 (512,)
    # NEG EX 3 (512,)
    # _____________________________________________________

    # WE UPSCALE with K.tile() so we have to remove the garbage. It's redundant.

    x1 = x1[0:BATCH_NUM_TRIPLETS * 3]

    anchor = x1[0:BATCH_NUM_TRIPLETS]
    positive_ex = x1[BATCH_NUM_TRIPLETS:2 * BATCH_NUM_TRIPLETS]
    negative_ex = x1[2 * BATCH_NUM_TRIPLETS:]

    sap = cosine_similarity(anchor, positive_ex)
    san = cosine_similarity(anchor, negative_ex)
    loss = K.mean(K.maximum(san - sap + alpha, 0.0))

    # we multiply x2 by 0 to have its gradient to be 0.
    # if we don't x2, its gradient is equal to None and it raises an error.
    # with our convention, we focus solely on x1 because the targets are given by the structure described above
    # with (anchor, positive examples, negative examples)
    return loss + 0 * x2
