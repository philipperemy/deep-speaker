
WORKING_DIR="/Users/premy/.cache"
AUDIO_DIR="/Users/premy/VCTK-Corpus-mini"

python cli.py build-audio-cache --audio_dir ${AUDIO_DIR} --working_dir ${WORKING_DIR} --parallel
python cli.py build-inputs-cache --audio_dir ${AUDIO_DIR} --working_dir ${WORKING_DIR}
python cli.py build-keras-inputs --data_filename /Users/premy/.cache/full_inputs.pkl --working_dir ${WORKING_DIR}
python cli.py train-model --working_dir ${WORKING_DIR}
