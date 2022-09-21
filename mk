#!/bin/bash

rm ./src/release/cuda_ed25519_vanity;
rm ./src/release/ecc_scan.o;
#export PATH=/usr/local/cuda/bin:$PATH;
make -j;
