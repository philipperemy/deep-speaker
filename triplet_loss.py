import keras.backend as K
from keras.layers import merge

alpha = 0.1


def triplet_loss(anchor, positive_ex, negative_ex):
    def cosine_similarity(x1, x2):
        return merge([x1, x2], mode='cos', concat_axis=-1)

    sap = cosine_similarity(anchor, positive_ex)
    san = cosine_similarity(anchor, negative_ex)

    loss = K.mean(K.abs(san - sap + alpha))
    return loss
