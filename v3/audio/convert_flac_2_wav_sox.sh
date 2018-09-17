#!/bin/bash

echo This script does not work
echo I left it here in case some of you could figure out why.
echo Use ffmpeg instead! It works well.
exit

folder=LibriSpeechSamples

for file in $(find "$folder" -type f -iname "*.flac")
do
    name=$(basename "$file" .flac)
    dir=$(dirname "$file")
    echo sox "$file" "$dir"/"$name".wav
    sox "$file" "$dir"/"$name".wav
done

# find $folder -name "*.flac" -exec rm -f {} \;

# https://unix.stackexchange.com/questions/341436/a-script-to-convert-flac-files-to-wav-is-not-working/341441