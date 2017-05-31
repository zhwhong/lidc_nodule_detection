#!/bin/bash
# This scripts downloads the ptb data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

mkdir -p data && cd data
wget --continue http://russellsstewart.com/s/tensorbox/inception_v1.ckpt
wget --continue http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
mkdir -p overfeat_rezoom && cd overfeat_rezoom
cd ..
echo "Extracting..."
tar xf resnet_v1_101_2016_08_28.tar.gz

if [[ "$1" == '--travis_tiny_data' ]]; then
    wget --continue http://russellsstewart.com/s/brainwash_tiny.tar.gz
    tar xf brainwash_tiny.tar.gz
    echo "Done."
else
    wget --continue http://russellsstewart.com/s/tensorbox/overfeat_rezoom/save.ckpt-150000v2
    wget --continue https://stacks.stanford.edu/file/druid:sx925dc9385/brainwash.tar.gz
    tar xf brainwash.tar.gz
    echo "Done."
fi
