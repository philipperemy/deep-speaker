#!/usr/bin/env bash

set -e

HOME_DIR=$(eval echo "~")
WORKING_DIR="/media/philippe/8TB/deep-speaker"
AUDIO_DIR="/media/philippe/8TB/datasets/libri2"

#python cli.py build-audio-cache --working_dir ${WORKING_DIR} --audio_dir ${AUDIO_DIR} --parallel
#python cli.py build-mfcc-cache --working_dir ${WORKING_DIR} --audio_dir ${AUDIO_DIR}
#python cli.py build-keras-inputs --working_dir ${WORKING_DIR}
#python cli.py train-model --working_dir ${WORKING_DIR} --pre_training_phase
python cli.py train-model --working_dir ${WORKING_DIR}


# axel -n 10 -a http://www.openslr.org/resources/12/train-clean-360.tar.gz
# tar xvzf train-clean-360.tar.gz
# LibriSpeech
#├── BOOKS.TXT
#├── CHAPTERS.TXT
#├── LICENSE.TXT
#├── README.TXT
#├── SPEAKERS.TXT
#└── train-clean-360
# ./flac2wav.sh LibriSpeech
# python cli.py libri-to-vctk-format --libri /media/philippe/8TB/datasets/libri --output /media/philippe/8TB/datasets/libri2
