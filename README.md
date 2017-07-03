# Deep Speaker from Baidu Research
Deep Speaker: an End-to-End Neural Speaker Embedding System https://arxiv.org/pdf/1705.02304.pdf

This project is still `WORK IN PROGRESS`!

Work accomplished so far:
- [x] Triplet loss
- [x] Model implementation
- [x] Data pipeline implementation. We're going to use the [LibriSpeech dataset](http://www.openslr.org/12/) with 5000+ different speakers.
- [ ] Train the models 

## Contributing

Please message me if you want to contribute. I'll be happy to hear your ideas. There are a lot of undisclosed things in the paper, such as:

- Input size to the network? Which inputs exactly?
- How many filter banks do we use?
- Sample Rate?

## LibriSpeech Dataset

Available here: http://www.openslr.org/12/

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
batch #0 processed in 97.44s, training loss = 0.7365261912345886.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #1 processed in 50.18s, training loss = 0.6806022524833679.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #2 processed in 51.11s, training loss = 0.6268694996833801.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #3 processed in 49.27s, training loss = 0.576310396194458.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #4 processed in 50.09s, training loss = 0.5292260050773621.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #5 processed in 48.77s, training loss = 0.48571109771728516.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #6 processed in 48.83s, training loss = 0.44574350118637085.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #7 processed in 47.99s, training loss = 0.40919357538223267.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #8 processed in 48.19s, training loss = 0.3759266138076782.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #9 processed in 47.99s, training loss = 0.3457900285720825.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #10 processed in 49.25s, training loss = 0.3185884952545166.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #11 processed in 46.92s, training loss = 0.29414060711860657.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #12 processed in 47.48s, training loss = 0.27225759625434875.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #13 processed in 47.80s, training loss = 0.2527392506599426.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #14 processed in 56.45s, training loss = 0.2354017049074173.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #15 processed in 55.14s, training loss = 0.22004079818725586.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
batch #16 processed in 49.47s, training loss = 0.20648227632045746.
x.shape = (6, 82, 32, 32, 3)
y.shape = (6,)
```
