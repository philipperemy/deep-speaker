import logging

import keras.backend as K

alpha = 0.2  # used in FaceNet https://arxiv.org/pdf/1503.03832.pdf


def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    dot = K.squeeze(K.batch_dot(x1, x2, axes=1), axis=1)
    logging.info('dot: {}'.format(dot))
    # as values have have length 1, we don't need to divide by norm (as it is 1)
    return dot


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
    # _____________________________________________________

    elements = int(K.int_shape(y_pred)[0] / 3)
    logging.info('elements={}'.format(elements))

    anchor = y_pred[0:elements]
    positive_ex = y_pred[elements:2 * elements]
    negative_ex = y_pred[2 * elements:]
    logging.info('anchor={}'.format(anchor))
    logging.info('positive_ex={}'.format(positive_ex))
    logging.info('negative_ex={}'.format(negative_ex))

    sap = batch_cosine_similarity(anchor, positive_ex)
    logging.info('sap={}'.format(sap))
    san = batch_cosine_similarity(anchor, negative_ex)
    logging.info('san={}'.format(san))
    loss = K.maximum(san - sap + alpha, 0.0)
    logging.info('loss={}'.format(loss))
    # total_loss = K.sum(loss)
    total_loss = K.mean(loss)
    logging.info('total_loss={}'.format(total_loss))
    return total_loss
