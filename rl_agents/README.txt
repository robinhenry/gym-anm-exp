How to train an agent
1. Modify `BASE_DIR` in `hyperparameters.py`
2. Run `script.sh` with {SAC, PPO} and <number of seeds>:
    conda activate anm
    ./script.sh SAC 3   # Trains 3 seeds (1..3) of SAC
