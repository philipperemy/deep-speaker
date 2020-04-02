#!/usr/bin/env bash

set -e

HOME_DIR=$(eval echo "~")
WORKING_DIR="${HOME_DIR}/.deep-speaker"
AUDIO_DIR="${HOME_DIR}/VCTK-Corpus-mini"

rm -rf ${WORKING_DIR}

python cli.py build-audio-cache --audio_dir ${AUDIO_DIR} --working_dir ${WORKING_DIR} --parallel
python cli.py build-inputs-cache --audio_dir ${AUDIO_DIR} --working_dir ${WORKING_DIR}
python cli.py build-keras-inputs --working_dir ${WORKING_DIR}
python cli.py train-model --working_dir ${WORKING_DIR}
