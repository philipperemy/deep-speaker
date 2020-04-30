import logging
from typing import List

import numpy as np
from tqdm import tqdm

from audio import Audio
from batcher import LazyTripletBatcher
from constants import NUM_FBANKS, NUM_FRAMES, BATCH_SIZE
from eval_metrics import evaluate
from models import ResCNNModel, select_model_class
from utils import score_fusion, embedding_fusion

logger = logging.getLogger(__name__)

EMBEDDING_FUSION = 0
SCORE_FUSION = 1


def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.multiply(x1, x2)
    s = np.sum(mul, axis=1)

    # l1 = np.sum(np.multiply(x1, x1),axis=1)
    # l2 = np.sum(np.multiply(x2, x2), axis=1)
    # as values have have length 1, we don't need to divide by norm (as it is 1)
    return s


def eval_models(working_dir: str, models: List[ResCNNModel], eval=evaluate):
    if isinstance(models, list) and len(models) > 1:  # multiple models -> fusion of results.
        y_pred_score_fusion = score_fusion(*[run_speaker_verification_task(working_dir, m) for m in models])
        y_pred_emb_fusion = run_speaker_verification_task(working_dir, models)
        assert y_pred_score_fusion.shape == y_pred_emb_fusion.shape
        y_true = np.zeros_like(y_pred_score_fusion)  # positive is at index 0.
        y_true[:, 0] = 1.0
        fm_1, tpr_1, acc_1, eer_1 = eval(y_pred_score_fusion, y_true)
        fm_2, tpr_2, acc_2, eer_2 = eval(y_pred_emb_fusion, y_true)
        logger.info(f'[score fusion] f-measure = {fm_1:.5f}, true positive rate = {tpr_1:.5f}, '
                    f'accuracy = {acc_1:.5f}, equal error rate = {eer_1:.5f}')
        logger.info(f'[emb fusion] f-measure = {fm_2:.5f}, true positive rate = {tpr_2:.5f}, '
                    f'accuracy = {acc_2:.5f}, equal error rate = {eer_2:.5f}')
    else:
        y_pred = run_speaker_verification_task(working_dir, models)
        y_true = np.zeros_like(y_pred)  # positive is at index 0.
        y_true[:, 0] = 1.0
        fm, tpr, acc, eer = eval(y_pred, y_true)
        logger.info(f'[single] f-measure = {fm:.5f}, true positive rate = {tpr:.5f}, '
                    f'accuracy = {acc:.5f}, equal error rate = {eer:.5f}')


def run_speaker_verification_task(working_dir, model):
    seed = 123
    embeddings_fusion_cond = isinstance(model, list)
    if embeddings_fusion_cond:
        assert len(model) == 2
    audio = Audio(working_dir)
    batcher = LazyTripletBatcher(working_dir, NUM_FRAMES, model=None)
    num_negative_speakers = 99
    num_speakers = len(audio.speaker_ids)
    y_pred = np.zeros(shape=(num_speakers, num_negative_speakers + 1))  # negatives + positive
    for i, positive_speaker in tqdm(enumerate(audio.speaker_ids), desc='test', total=num_speakers):
        # convention id[0] is anchor speaker, id[1] is positive, id[2:] are negative.
        input_data = batcher.get_speaker_verification_data(positive_speaker, num_negative_speakers, seed=i * seed)
        input_data_2 = batcher.get_speaker_verification_data(positive_speaker, num_negative_speakers, seed=i * seed)
        np.testing.assert_array_equal(input_data, input_data_2)
        # batch size is not relevant. just making sure we don't push too much on the GPU.
        if embeddings_fusion_cond:
            predictions = embedding_fusion(model[0].m.predict(input_data, batch_size=BATCH_SIZE),
                                           model[1].m.predict(input_data, batch_size=BATCH_SIZE))
        else:
            predictions = model.m.predict(input_data, batch_size=BATCH_SIZE)
        anchor_embedding = predictions[0]
        for j, other_than_anchor_embedding in enumerate(predictions[1:]):  # positive + negatives
            y_pred[i][j] = batch_cosine_similarity([anchor_embedding], [other_than_anchor_embedding])[0]
    return y_pred


def test(working_dir, model_names: tuple, checkpoint_files: tuple):
    batch_input_shape = [None, NUM_FRAMES, NUM_FBANKS, 1]
    models = []
    for checkpoint_file, model_name in zip(checkpoint_files, model_names):
        dsm = select_model_class(model_name)(batch_input_shape)
        if checkpoint_file is not None:
            logger.info(f'Found checkpoint [{checkpoint_file}] for [{model_name}]. Loading weights...')
            dsm.m.load_weights(checkpoint_file, by_name=True)
        else:
            logger.info(f'Could not find any checkpoint in {checkpoint_file}.')
            exit(1)
        models.append(dsm)
    eval_models(working_dir, models)
    if len(models) > 1:
        for model in models:
            eval_models(working_dir, model)
