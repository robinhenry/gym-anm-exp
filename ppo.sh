#!/usr/bin/env bash

python3 run.py ppo  --env anm \
                    --n_envs 8 \
                    --seed 2020 \
                    --num-episodes 100000 \
                    -T 1000 \
                    --eval-period 10 \
                    --num_rollouts 20 \
                    --name first_run
