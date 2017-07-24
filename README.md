# Deep Speaker from Baidu Research
[![license](https://img.shields.io/badge/License-Apache_2.0-brightgreen.svg)](https://github.com/philipperemy/keras-attention-mechanism/blob/master/LICENSE) 
[![dep2](https://img.shields.io/badge/Keras-2.0+-brightgreen.svg)](https://keras.io/) 
[![dep1](https://img.shields.io/badge/Status-Work_In_Progress-orange.svg)](https://www.tensorflow.org/) 

Deep Speaker: an End-to-End Neural Speaker Embedding System https://arxiv.org/pdf/1705.02304.pdf

Work accomplished so far:
- [x] Triplet loss
- [x] Triplet loss test
- [x] Model implementation
- [x] Data pipeline implementation. We're going to use the [LibriSpeech dataset](http://www.openslr.org/12/) with 2300+ different speakers.
- [ ] Train the models 

<p align="center">
  <img src="assets/1.png" width="400">
  <br><i>Visualization of a possible triplet (Anchor, Positive, Negative) in the cosine similarity space</i>
</p>

## Contributing

Please message me if you want to contribute. I'll be happy to hear your ideas. There are a lot of undisclosed things in the paper, such as:

- Input size to the network? Which inputs exactly?
- How many filter banks do we use?
- Sample Rate?

## LibriSpeech Dataset

Available here: http://www.openslr.org/12/

List of possible other datasets: http://kaldi-asr.org/doc/examples.html

Extract of this dataset:

```
                                                                            filenames chapter_id speaker_id dataset_id
0  /Volumes/Transcend/data-set/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.wav     128104       1272  dev-clean
1  /Volumes/Transcend/data-set/LibriSpeech/dev-clean/1272/128104/1272-128104-0001.wav     128104       1272  dev-clean
2  /Volumes/Transcend/data-set/LibriSpeech/dev-clean/1272/128104/1272-128104-0002.wav     128104       1272  dev-clean
3  /Volumes/Transcend/data-set/LibriSpeech/dev-clean/1272/128104/1272-128104-0003.wav     128104       1272  dev-clean
4  /Volumes/Transcend/data-set/LibriSpeech/dev-clean/1272/128104/1272-128104-0004.wav     128104       1272  dev-clean
5  /Volumes/Transcend/data-set/LibriSpeech/dev-clean/1272/128104/1272-128104-0005.wav     128104       1272  dev-clean
6  /Volumes/Transcend/data-set/LibriSpeech/dev-clean/1272/128104/1272-128104-0006.wav     128104       1272  dev-clean
7  /Volumes/Transcend/data-set/LibriSpeech/dev-clean/1272/128104/1272-128104-0007.wav     128104       1272  dev-clean
8  /Volumes/Transcend/data-set/LibriSpeech/dev-clean/1272/128104/1272-128104-0008.wav     128104       1272  dev-clean
9  /Volumes/Transcend/data-set/LibriSpeech/dev-clean/1272/128104/1272-128104-0009.wav     128104       1272  dev-clean
```

## Training example on CPU

```
batch #0 processed in 233.63s, training loss = 0.7363479137420654.
batch #1 processed in 176.31s, training loss = 0.6804276704788208.
batch #2 processed in 178.94s, training loss = 0.6266987919807434.
batch #3 processed in 172.45s, training loss = 0.5761439204216003.
batch #4 processed in 164.76s, training loss = 0.52906334400177.
batch #5 processed in 165.07s, training loss = 0.4855523407459259.
batch #6 processed in 156.74s, training loss = 0.4455888271331787.
batch #7 processed in 145.81s, training loss = 0.40904250741004944.
batch #8 processed in 141.85s, training loss = 0.3757786452770233.
batch #9 processed in 142.46s, training loss = 0.34564515948295593.
batch #10 processed in 157.34s, training loss = 0.3184462785720825.
batch #11 processed in 183.19s, training loss = 0.29400110244750977.
batch #12 processed in 151.84s, training loss = 0.27212029695510864.
```
