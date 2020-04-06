import keras.backend as K

ALPHA = 0.2  # used in FaceNet https://arxiv.org/pdf/1503.03832.pdf


def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    dot = K.squeeze(K.batch_dot(x1, x2, axes=1), axis=1)
    # as values have have length 1, we don't need to divide by norm (as it is 1)
    return dot


def deep_speaker_loss(y_true, y_pred):
    # y_true is not used. we respect this convention:
    # y_true.shape = (batch_size, embedding_size)
    # y_pred.shape = (batch_size, embedding_size)
    # EXAMPLE:
    # _____________________________________________________
    # ANCHOR 1 (512,)
    # ANCHOR 2 (512,)
    # POS EX 1 (512,)
    # POS EX 2 (512,)
    # NEG EX 1 (512,)
    # NEG EX 2 (512,)
    # _____________________________________________________
    split = K.shape(y_pred)[0] // 3

    anchor = y_pred[0:split]
    positive_ex = y_pred[split:2 * split]
    negative_ex = y_pred[2 * split:]

    sap = batch_cosine_similarity(anchor, positive_ex)
    san = batch_cosine_similarity(anchor, negative_ex)
    loss = K.maximum(san - sap + ALPHA, 0.0)
    total_loss = K.sum(loss)
    return total_loss
