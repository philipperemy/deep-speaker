import logging

import numpy as np

from batcher import KerasConverter, TripletBatcher
from constants import NUM_FBANKS, NUM_FRAMES, CHECKPOINTS_TRIPLET_DIR, BATCH_SIZE
from conv_models import DeepSpeakerModel
from triplet_loss import deep_speaker_loss
from utils import load_best_checkpoint, enable_deterministic

logger = logging.getLogger(__name__)


def eval_model(working_dir: str, model: DeepSpeakerModel):
    enable_deterministic()
    kc = KerasConverter(working_dir)
    batcher = TripletBatcher(kc.kx_train, kc.ky_train, kc.kx_test, kc.ky_test)
    test_loss = []
    while True:
        x, y = batcher.get_batch(BATCH_SIZE, is_test=True)
        print('sum(x^2)', np.mean(x ** 2))
        # p = model.m.predict(x)
        test_loss.append(model.m.evaluate(x, y, verbose=0, batch_size=BATCH_SIZE))
        print(np.mean(test_loss), len(test_loss))


def test2(working_dir, checkpoint_file=None):
    batch_input_shape = [None, NUM_FRAMES, NUM_FBANKS, 1]
    dsm = DeepSpeakerModel(batch_input_shape)
    dsm.m.compile(optimizer='adam', loss=deep_speaker_loss)
    if checkpoint_file is None:
        checkpoint_file = load_best_checkpoint(CHECKPOINTS_TRIPLET_DIR)
    if checkpoint_file is not None:
        logger.info(f'Found checkpoint [{checkpoint_file}]. Loading weights...')
        dsm.m.load_weights(checkpoint_file, by_name=True)
    else:
        logger.info(f'Could not find any checkpoint in {checkpoint_file}.')
        exit(1)

    eval_model(working_dir, model=dsm)
