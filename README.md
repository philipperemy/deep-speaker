# Deep Speaker
*On VCTK Corpus with ~110 speakers*

This is the first version (v1) that works correctly with a dataset of ~110 speakers and a smaller model (~100k parameters).

## Get Started

First of all, be sure to have at least 16GB of memory before running those steps. At the moment, everything is loaded in memory at the beginning for faster training speed. For now a GPU is not required because the models are pretty small. It will actually run faster on a CPU I guess.
I could run everything on my MacBookPro 2017.


Now let's clone the repository, create a virtual environment, install the dependencies, build the audio and inputs caches, run the softmax pre-training and start the training of the deep speaker embeddings.

### Installation

```
git clone git@github.com:philipperemy/deep-speaker.git

DS_DIR=~/deep-speaker-data
AUDIO_DIR=$DS_DIR/VCTK-Corpus/
CACHE_DIR=$DS_DIR/cache/

mkdir -p $DS_DIR

./download_vctk.sh
mv ~/VCTK-Corpus $DS_DIR

virtualenv -p python3.6 $DS_DIR/venv-speaker # probably will work on every python3 impl (e.g. 3.5).
source $DS_DIR/venv-speaker/bin/activate

pip install -r requirements.txt
pip install tensorflow # or tensorflow-gpu
```

### Generate audio caches

```
# 9min with i7-8770K
python cli.py --regenerate_full_cache --multi_threading --cache_output_dir $CACHE_DIR --audio_dir $AUDIO_DIR

# 13min with i7-8770K
python cli.py --generate_training_inputs --multi_threading --cache_output_dir $CACHE_DIR --audio_dir $AUDIO_DIR
```

### Run softmax pre-training and embeddings training with triplet loss

```
python train_cli.py --loss_on_softmax --data_filename $CACHE_DIR/full_inputs.pkl
python train_cli.py --loss_on_embeddings --normalize_embeddings --data_filename $CACHE_DIR/full_inputs.pkl
python train_cli.py --loss_on_softmax --freeze_embedding_weights --normalize_embeddings --data_filename $CACHE_DIR/full_inputs.pkl
python cli.py --update_cache --multi_threading --audio_dir $NEW_AUDIO_DIR --cache_output_dir $CACHE_DIR
```

### Generate embeddings with a pre-trained network

#### From speakers in the dataset

```
python cli.py --unseen_speakers p362,p363 --audio_dir $AUDIO_DIR --cache_output_dir $CACHE_DIR
```


Then let's pick up two speakers from the out sample set (never seen from the training steps).

- We first check that the embeddings are L2-normalized
- We then check that the SAP is much lower compared to SAN.

```
SAP = 0.016340159318026376 (cosine distance p363 to p363 - same speaker)
SAN = 0.7578228781188744 (cosine distance p363 to p362 - different speaker)
```


#### From any WAV files

```
NEW_AUDIO_DIR=./samples/PhilippeRemy/
python cli.py --update_cache --multi_threading --audio_dir $NEW_AUDIO_DIR --cache_output_dir $CACHE_DIR


python cli.py --unseen_speakers PhilippeRemy,PhilippeRemy --audio_dir $NEW_AUDIO_DIR --cache_output_dir $CACHE_DIR
python cli.py --unseen_speakers p225,PhilippeRemy --audio_dir $NEW_AUDIO_DIR --cache_output_dir $CACHE_DIR
```

### Miscellaneous

Once the model is trained, we can freeze the weights and re-train your softmax to see if the embeddings we got make sense. Accuracy is around 71%. Not bad!

```
python train_triplet_softmax_model.py --loss_on_softmax --freeze_embedding_weights --normalize_embeddings
```

### Training comments

- After the softmax pre-training, the speaker classification accuracy should be around 95%.
- Training the embeddings with the triplet loss (specific to deep speaker) takes time and the loss should go around 0.01-0.02 after ~5k steps (on un-normalized embeddings). After only 2k steps, I had 0.04-0.05. I noticed that the softmax pre-training really helped the convergence be faster. The case where (anchor speaker == positive speaker == negative speaker) yields a loss of 0.20. This optimizer gets stuck and cannot do much. This is expected. We can clearly see that the model is learning something. I recall that we train with (anchor speaker == positive speaker != negative speaker).
- Then we re-train again the softmax layer with the new embeddings. We freeze them and we look at the new classification accuracy. It's now around 71%. We expect it to be less than 95% of course, because the embeddings are not trained to maximize the classification accuracy but to reduce the triplet loss (maximize cosine similarity between different speakers).
- At the moment, I'm using a sigmoid for the embeddings. Meaning that the embeddings are defined on [0, 1]^n. Using tanh will project them on [-1, 1]^n.
