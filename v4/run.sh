#!/usr/bin/env bash

set -e

HOME_DIR=$(eval echo "~")
WORKING_DIR="${HOME_DIR}/.deep-speaker"
AUDIO_DIR="${HOME_DIR}/VCTK-Corpus"

#python cli.py build-audio-cache --working_dir ${WORKING_DIR} --audio_dir ${AUDIO_DIR} --parallel
#python cli.py build-mfcc-cache --working_dir ${WORKING_DIR} --audio_dir ${AUDIO_DIR}
python cli.py build-keras-inputs --working_dir ${WORKING_DIR}
python cli.py train-model --working_dir ${WORKING_DIR} --pre_training_phase
python cli.py train-model --working_dir ${WORKING_DIR}
