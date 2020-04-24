#!/usr/bin/env bash

set -e

HOME_DIR=$(eval echo "~")
WORKING_DIR="${HOME_DIR}/deep-speaker"
AUDIO_DIR="${HOME_DIR}/VCTK-Corpus-mini"

python cli.py build-mfcc-cache --working_dir "${WORKING_DIR}" --audio_dir "${AUDIO_DIR}"
python cli.py build-keras-inputs --working_dir "${WORKING_DIR}"
python cli.py train-model --working_dir "${WORKING_DIR}" --pre_training_phase
python cli.py train-model --working_dir "${WORKING_DIR}"

# axel -n 10 -a http://www.openslr.org/resources/12/train-clean-360.tar.gz
# tar xvzf train-clean-360.tar.gz
# LibriSpeech
#├── BOOKS.TXT
#├── CHAPTERS.TXT
#├── LICENSE.TXT
#├── README.TXT
#├── SPEAKERS.TXT
#└── train-clean-360
# Flac or Wav does not matter. It will be the same audio array.
# python cli.py libri-to-vctk-format --libri /media/philippe/8TB/datasets/libri/LibriSpeech --output /media/philippe/8TB/datasets/libri-vctk-format-train-clean360 --subset train-clean-360
# python cli.py libri-to-vctk-format --libri /media/philippe/8TB/datasets/libri/LibriSpeech --output /media/philippe/8TB/datasets/libri-vctk-format
