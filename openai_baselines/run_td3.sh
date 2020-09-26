#!/usr/bin/env bash

# Train TD3 agent in different threads.
N_RUNS=10
INIT_SEED=2020
TIMESTEPS=5000000
NICE_VALUE=10

for i in $(seq 1 $N_RUNS)
do
  seed=$($INIT_SEED + $i)
  screen -dmS "seed$seed" conda activate anm; nice -n $NICE_VALUE python td3.py -s $seed -T $TIMESTEPS
done
