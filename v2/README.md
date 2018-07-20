# Deep Speaker V2 (on a smaller scale for now)

- Using VCTK Corpus
- Using the codebase from [speaker-change-detection](https://github.com/philipperemy/speaker-change-detection)

We start on a smaller dataset (109 speakers) and on much smaller models (~100k parameters).

## Get Started

We're going to use pre-processed files for the training and the inference. Because it takes a very long time to generate cache and inputs (~2 hours), I packaged them and uploaded them here:

- Cache uploaded at [cache-speaker-change-detection.zip](https://drive.google.com/open?id=1NRBBE7S1ecpbXQBfIyhY9O1DDNsBc0my)  (unzip it in `/tmp/`)
- [speaker-change-detection-data.pkl](https://drive.google.com/open?id=12gMYaV-ymQOtkYHCf9HxPurb9vB6dADK) (place it in `/tmp/`)
- [speaker-change-detection-norm.pkl](https://drive.google.com/open?id=1vykyS3bxKbkuhGtk36eTWfW9ZkqwJi6e) (place it in `/tmp/`)

After doing this, those commands should work:

- `ls -l /tmp/speaker-change-detection-data.pkl`
- `ls -l /tmp/speaker-change-detection-norm.pkl`
- `ls -l /tmp/speaker-change-detection/*.pkl`

Now let's clone the repository, create a virtual environment, install the dependencies, run the softmax pre-training and start the training of the deep speaker embeddings.

```
git clone git@github.com:philipperemy/deep-speaker.git
cd deep-speaker/v2
virtualenv -p python3.6 venv # probably will work on every python3 impl.
source venv/bin/activate
pip install -r requirements.txt
# download the cache and all the files specified above (you can re-generate them yourself if you wish, it just takes ~2 hours).
cd ml/
export PYTHONPATH=..:$PYTHONPATH; python 0_generate_inputs.py
export PYTHONPATH=..:$PYTHONPATH; python 1_train_triplet_softmax_model.py --loss_on_softmax # softmax pre-training
export PYTHONPATH=..:$PYTHONPATH; python 1_train_triplet_softmax_model.py --loss_on_embeddings
```

## Comments

- After the softmax pre-training, the speaker classification accuracy should be around 95%.
- Training the embeddings with the triplet loss (specific to deep speaker) takes time and should be around 0.01~0.02 after 150k steps. The case where anchor speaker == positive speaker == negative speaker yields a loss of 0.20. This optimizer gets stuck and cannot do much. This is expected. We can clearly see that the model is learning something.
- Then we re-train again the softmax layer with the new embeddings. We freeze them and we look at the new classification accuracy. It's now around 65%. We expect it to be less than 95% of course. Because the embeddings are not trained to maximize the classification accuracy but to reduce the triplet loss (maximize cosine similarity between different speakers).
