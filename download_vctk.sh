#!/usr/bin/env bash
endpoint="http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz"
if [ -z "$(command -v axel)" ]
then
  wget ${endpoint}
else
  axel -n 10 ${endpoint} # faster download.
fi
 tar xvzf VCTK-Corpus.tar.gz -C ~