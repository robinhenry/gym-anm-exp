import gym
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt


# Read name of file from command line.
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algo', type=str, help='the name of the algorithm')
parser.add_argument('-s', '--seed', type=int, help='the random seed')
args = parser.parse_args()

# Load run data.
log_path = 'results/' + args.algo + '_' + args.seed + '/evaluations.npz'
results = np.load(log_path)
timesteps = results['timesteps']
ep_lengths = results['ep_lengths']
results = results['results']

print(results[:5])
print(ep_lengths[:5])
print(timesteps[:5])

# Plot run data.
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(timesteps, results)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Return')
