#!/bin/bash
# This script trains a number of RL agents,
# each in a different thread with a different random
# seed.
# For example, the following will train 3 SAC agents:
#   $ ./run.sh SAC 3

# Check which Python executable is being used.
echo "Running in env:"
which python

# Run for random seeds [0..$2].
for seed in $(seq "$2")
do
   python train.py $1 -s $seed &
done
