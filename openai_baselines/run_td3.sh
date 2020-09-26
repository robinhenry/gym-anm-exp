#!/usr/bin/env bash

# Train TD3 agent in different threads.
N_RUNS=5
INIT_SEED=2020
TIMESTEPS=5000000
NICE_VALUE=10

# Activate conda env.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anm
echo -e "using Python environment: \c"
echo | which python

for i in $(seq 1 $N_RUNS)
do
  seed=$((INIT_SEED + i))
  log_file="./logs/td3_$seed.txt"
  screen -dmS "seed$seed" bash -c 'nice -n $NICE_VALUE python td3.py -s $seed -T $TIMESTEPS > $log_file 2>&1';
done
