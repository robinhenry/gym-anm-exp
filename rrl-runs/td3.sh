#!/usr/bin/env bash

nice -n 10 python3 run.py td3 --env anm \
                          --seed 2020 \
                          --num-episodes 1000 \
                          -T 1000 \
                          --eval-period 10 \
                          --num_rollouts 20 \
                          --name first_run
