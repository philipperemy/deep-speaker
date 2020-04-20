#!/bin/bash

set -e

# axel or wget.
axel -n 10 -a http://www.openslr.org/resources/12/dev-clean.tar.gz
axel -n 10 -a http://www.openslr.org/resources/12/dev-other.tar.gz
axel -n 10 -a http://www.openslr.org/resources/12/test-clean.tar.gz
axel -n 10 -a http://www.openslr.org/resources/12/test-other.tar.gz
axel -n 10 -a http://www.openslr.org/resources/12/train-clean-100.tar.gz
axel -n 10 -a http://www.openslr.org/resources/12/train-clean-360.tar.gz
axel -n 10 -a http://www.openslr.org/resources/12/train-other-500.tar.gz

tar xvzf dev-clean.tar.gz
tar xvzf dev-other.tar.gz
tar xvzf test-clean.tar.gz
tar xvzf test-other.tar.gz
tar xvzf train-clean-100.tar.gz
tar xvzf train-clean-360.tar.gz
tar xvzf train-other-500.tar.gz

# LibriSpeech/train-clean-360*
# LibriSpeech/train-other-500*
