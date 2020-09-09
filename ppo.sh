#!/usr/bin/env bash

which python
python3 run.py ppo --env pendulum-v0 \
                    --n_envs 10 \
                    --seed 2020 \
                    --num-episodes 100000 \
                    -T 1000 \
                    --eval-period 10 \
                    --num_rollouts 20 \
                    --name first_run
