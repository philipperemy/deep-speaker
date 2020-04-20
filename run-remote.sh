#!/usr/bin/env bash

set -e

WORKING_DIR="/media/philippe/8TB/ds-test"
PRE_TRAINING_WORKING_DIR="${WORKING_DIR}/pre-training"
TRIPLET_TRAINING_WORKING_DIR="${WORKING_DIR}/triplet-training"

mkdir -p "${WORKING_DIR}"

# Download and extract the LibriSpeech dataset.
cp download_librispeech.sh "${WORKING_DIR}"
cd "${WORKING_DIR}" && bash download_librispeech.sh && cd -

# Build MFCC caches for pre-training and triplet-training.
python cli.py build-mfcc-cache --working_dir "${PRE_TRAINING_WORKING_DIR}" --audio_dir "${WORKING_DIR}/LibriSpeech/train-clean-360"
python cli.py build-mfcc-cache --working_dir "${TRIPLET_TRAINING_WORKING_DIR}" --audio_dir "${WORKING_DIR}/LibriSpeech"

# Build the Keras inputs for the pre-training.
# python cli.py build-keras-inputs --working_dir "${PRE_TRAINING_WORKING_DIR}"

# Pre-training (0.92k speakers).
# python cli.py train-model --working_dir "${PRE_TRAINING_WORKING_DIR}" --pre_training_phase

# Triplet-training (2.48k speakers).
# python cli.py train-model --working_dir "${TRIPLET_TRAINING_WORKING_DIR}"
