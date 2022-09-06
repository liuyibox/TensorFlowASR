#!/bin/bash
rm -rf dataset
mkdir dataset
cd dataset

wget https://us.openslr.org/resources/12/train-clean-100.tar.gz
wget https://us.openslr.org/resources/12/test-clean.tar.gz
wget https://us.openslr.org/resources/12/dev-clean.tar.gz

tar -xzvf train-clean-100.tar.gz
tar -xzvf dev-clean.tar.gz
tar -xzvf test-clean.tar.gz
