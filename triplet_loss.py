import logging

import keras.backend as K

alpha = 0.1


# K.sum(anchor * positive_ex, axis=1).eval() == K.batch_dot(anchor, positive_ex, axes=1)

def batch_norm(x):
    return K.square(K.sum(K.squeeze(K.batch_dot(x, x, axes=1), 1)))


def cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    return K.squeeze(K.batch_dot(x1, x2, axes=1), axis=1) / (batch_norm(x1) * batch_norm(x2))


def deep_speaker_loss(y_true, y_pred):
    logging.info('y_true={}'.format(y_true))
    logging.info('y_pred={}'.format(y_pred))
    # y_true.shape = (batch_size, embedding_size)
    # y_pred.shape = (batch_size, embedding_size)
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

    elements = int(K.int_shape(y_pred)[0] / 3)
    logging.info('elements={}'.format(elements))

    anchor = y_pred[0:elements]
    positive_ex = y_pred[elements:2 * elements]
    negative_ex = y_pred[2 * elements:]
    logging.info('anchor={}'.format(anchor))
    logging.info('positive_ex={}'.format(positive_ex))
    logging.info('negative_ex={}'.format(negative_ex))

    sap = cosine_similarity(anchor, positive_ex)
    logging.info('sap={}'.format(sap))
    san = cosine_similarity(anchor, negative_ex)
    logging.info('san={}'.format(san))
    loss = K.sum(K.maximum(san - sap + alpha, 0.0))
    logging.info('loss={}'.format(loss))

    return loss
