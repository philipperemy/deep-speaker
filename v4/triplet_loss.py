import keras.backend as K

# ALPHA = 0.2  # used in FaceNet https://arxiv.org/pdf/1503.03832.pdf
ALPHA = 0.1  # used in Deep Speaker.


def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    dot = K.squeeze(K.batch_dot(x1, x2, axes=1), axis=1)
    # as values have have length 1, we don't need to divide by norm (as it is 1)
    return dot


def deep_speaker_loss(y_true, y_pred):
    # y_true is not used. we respect this convention:
    # y_true.shape = (batch_size, embedding_size) [not used]
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

    # If the loss does not decrease below ALPHA then the model does not learn anything.
    # If all anchor = positive = negative (model outputs the same vector always).
    # Then sap = san = 1. and loss = max(alpha,0) = alpha.
    # On the contrary if anchor = positive = [1] and negative = [-1].
    # Then sap = 1 and san = -1. loss = max(-1-1+0.1,0) = max(-1.9, 0) = 0.
    sap = batch_cosine_similarity(anchor, positive_ex)
    san = batch_cosine_similarity(anchor, negative_ex)
    loss = K.maximum(san - sap + ALPHA, 0.0)
    total_loss = K.mean(loss)
    return total_loss


if __name__ == '__main__':
    import numpy as np

    y = np.random.uniform(low=-1, high=1, size=(3 * 4, 10))
    y /= np.linalg.norm(y, 2, axis=1, keepdims=True)
    deep_speaker_loss(y_true=y, y_pred=y)
