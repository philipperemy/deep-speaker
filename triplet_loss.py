import keras.backend as K
from keras import losses

from constants import BATCH_SIZE

alpha = 0.1


def deep_speaker_loss(x1, x2):
    # x1.shape = (batch_size, embedding_size)
    # x2.shape = (batch_size, embedding_size)
    # CONVENTION IS:
    # FIRST THIRD IS ANCHOR (SP1)
    # SECOND THIRD IS POSITIVE EXAMPLE (SP1)
    # LAST THIRD IS NEGATIVE EXAMPLE (SP2)
    # WITH THIS CONVENTION WE ACTUALLY DON'T USE THE TARGETS.
    one_third_of_batch_size = BATCH_SIZE // 3
    anchor = x1[0:one_third_of_batch_size]
    positive_ex = x1[one_third_of_batch_size:2 * one_third_of_batch_size]
    negative_ex = x1[2 * one_third_of_batch_size:]
    return triplet_loss(anchor, positive_ex, negative_ex)


def triplet_loss(anchor, positive_ex, negative_ex):
    # def cosine_similarity(x1, x2):
    #    return merge([x1, x2], mode='cos', concat_axis=-1)

    sap = losses.cosine_proximity(anchor, positive_ex)
    san = losses.cosine_proximity(anchor, negative_ex)

    loss = K.mean(K.abs(san - sap + alpha))
    return loss
