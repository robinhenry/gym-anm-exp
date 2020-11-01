import gym
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import os


# Read name of file from command line.
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algo', type=str, help='the name of the algorithm')
parser.add_argument('-s', '--seed', type=int, help='the random seed')
parser.add_argument('-e', '--env', type=str, help='the environment')
args = parser.parse_args()

# Load run data.
log_folder = 'results/' + args.env + '_' + args.algo + '_' + str(args.seed) + '/'
results = np.load(log_folder + 'evaluations.npz')
timesteps = results['timesteps']
ep_lengths = results['ep_lengths']
results = results['results']

# Mean and std of returns.
mean = np.mean(results, axis=1).flatten()
std = np.std(results, axis=1).flatten()

# Plot run data.
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(timesteps, mean)
ax.fill_between(timesteps, mean - std, mean + std, alpha=0.4)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Return')

# Save and close the figure.
os.makedirs('plots', exist_ok=True)
figure_path = 'plots/' + args.env + '_' + args.algo + '_' + str(args.seed) + '.png'
fig.savefig(figure_path)
plt.close(fig)
