#!/usr/bin/env bash

#!/usr/bin/env bash

cd ..
python3 robin/run.py ddpg --env anm \
                          --seed 2020 \
                          --num-episodes 100 \
                          -T 1000 \
                          --eval-period 10 \
                          --num_rollouts 20 \
                          --name first_run
