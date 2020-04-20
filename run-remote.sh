#!/usr/bin/env bash

set -e

WORKING_DIR="/media/philippe/8TB/deep-speaker"
PRE_TRAINING_WORKING_DIR="${WORKING_DIR}/pre-training"
TRIPLET_TRAINING_WORKING_DIR="${WORKING_DIR}/triplet-training"

# WORKING_DIR/LibriSpeech
cd "${WORKING_DIR}" && bash download_librispeech.sh && cd -

# LIBRI_SPEECH_DATASET="${WORKING_DIR}/LibriSpeech"
#python cli.py libri-to-vctk-format --libri "${LIBRI_SPEECH_DATASET}" --output "${PRE_TRAINING_WORKING_DIR}/audio" --subset train-clean-360
#python cli.py libri-to-vctk-format --libri "${LIBRI_SPEECH_DATASET}" --output "${TRIPLET_TRAINING_WORKING_DIR}/audio"

# Pre-training (0.92k speakers).
python cli.py build-mfcc-cache --working_dir "${PRE_TRAINING_WORKING_DIR}" --audio_dir "${WORKING_DIR}/LibriSpeech/train-clean-360"
python cli.py build-keras-inputs --working_dir "${PRE_TRAINING_WORKING_DIR}"
python cli.py train-model --working_dir "${PRE_TRAINING_WORKING_DIR}" --pre_training_phase

# Triplet-training (2.48k speakers).
python cli.py build-mfcc-cache --working_dir "${TRIPLET_TRAINING_WORKING_DIR}" --audio_dir "${WORKING_DIR}/LibriSpeech"
python cli.py train-model --working_dir "${TRIPLET_TRAINING_WORKING_DIR}"
