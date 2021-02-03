import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.monitor import load_results

from analyse_results.plt_setup import set_rc_params

set_rc_params()
FIGSIZE = (10, 4)

# Baseline best discounted rewards.
dc_opf_best = -11.038
mpc_best = -6.369

# Agent result folders.
folder = './results/gym_anm:ANM6Easy-v0/'
agents = ['ppo', 'sac']
run_id = '1'

# Load training stats.
eval_stats = {}
training_stats = {}
for agent in agents:
    dir =  folder + agent + '_' + run_id + '/'

    # evaluations.npz
    f = dir + 'evaluations.npz'
    d = np.load(f)
    timesteps = d['timesteps']
    results = d['results'][:, 0, 0]
    discounted_results = d['discounted_results'][:, 0, 0]
    ep_lengths = d['ep_lengths'][:, 0]
    eval_stats[agent] = (timesteps, results, discounted_results, ep_lengths)

    # monitor.csv
    training_stats[agent] = load_results(dir)

# Plot discounted returns.
fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for agent, d in eval_stats.items():
    ax.plot(d[0] / 1e6, d[2], label=agent.upper())
    print(f'Best for {agent}: ', np.max(d[2]))

ax.set_xlim([None, 4])
ax.set_xlabel('Timestep (million)')
ax.set_ylabel('Discounted return')
ax.legend(loc='lower right', fontsize='small')

fig.savefig('figures/discounted_return.pdf', bbox_inches='tight')

# Plot discounted returns zoomed-in with baseline performances.
fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for agent, d in eval_stats.items():
    ax.plot(d[0] / 1e6, d[2], label=agent.upper())
    print(f'Best for {agent}: ', np.max(d[2]))
ax.hlines(dc_opf_best, 0, 4, colors='g', linestyles='dashed', label=r'$\pi_{MPC-16}^{constant}$')
ax.hlines(mpc_best, 0, 4, colors='r', linestyles='dashed', label=r'$\pi_{MPC-128}^{forecast}$')

ax.set_xlim([None, 4])
ax.set_ylim([-200, 0])
ax.set_xlabel('Timestep (million)')
ax.set_ylabel('Discounted return')
ax.legend(loc='lower right', fontsize='small')

fig.savefig('figures/discounted_return_zoom.pdf', bbox_inches='tight')

# Plot total rewards.
fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for agent, d in eval_stats.items():
    ax.plot(d[0] / 1e6, d[1], label=agent.upper())
    print(f'Best for {agent}: ', np.max(d[1]))

ax.set_xlim([None, 4])
ax.set_xlabel('Timestep (million)')
ax.set_ylabel('Non-discounted return (T=2000)')
ax.legend(loc='lower right')

fig.savefig('figures/nondiscounted_return.pdf', bbox_inches='tight')

# Plot the evolution of episode length over training.
fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for agent, d in training_stats.items():
    ax.plot(d['l'], label=agent.upper())
    print(f'Longest episode for {agent}: ', np.max(d['l']))

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Training episode')
ax.set_ylabel('Episode length')
ax.legend(loc='lower right')

fig.savefig('figures/training_ep_length.pdf', bbox_inches='tight')


# plt.show()

if __name__ == '__main__':
    print('Done!')