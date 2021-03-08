import matplotlib.pyplot as plt
import numpy as np
import os

from analyze_results.plt_setup import set_rc_params

set_rc_params()
FIGSIZE = (10, 4)

# Baseline best discounted rewards.
dc_opf_best = -11.038
mpc_best = -6.369

# Agent result folders.
folder = '/Users/uni/Dev/gym-anm_exps/analyze_results/results/'
agent_groups = [['sac', 'sac2'], ['ppo']]

# Load training stats.
eval_stats = {a: [] for a in [group[0] for group in agent_groups]}
training_stats = {a: [] for a in [group[0] for group in agent_groups]}
for agent_group in agent_groups:
    agent = agent_group[0]

    for run in os.listdir(os.path.join(folder, agent)):
        if not run.startswith('run_'):
            continue

        timesteps, results, discounted_results, ep_lengths = [], [], [], []
        for a in agent_group:
            f = os.path.join(folder, a, run, 'evaluations.npz')
            d = np.load(f)
            timesteps.append(d['timesteps'])
            results.append(d['results'][:, 0, 0])
            discounted_results.append(d['discounted_results'][:, 0, 0])
            ep_lengths.append(d['ep_lengths'][:, 0])

        # Concatenate over multiple training sessions.
        timesteps = np.hstack(timesteps)
        results = np.hstack(results)
        discounted_results = np.hstack(discounted_results)
        ep_lengths = np.hstack(ep_lengths)

        d = np.array([timesteps, results, discounted_results, ep_lengths])
        eval_stats[agent].append(d)

        # monitor.csv
        # training_stats[agent].append(load_results(subdir))

# Clean statistics.
for agent, d in eval_stats.items():
    t_min = np.min([x.shape[1] for x in d])
    eval_stats[agent] = np.stack([x[:, :t_min] for x in d])

# for agent, d in training_stats.items():
#     t_min = np.min([x.shape[0] for x in d])
#     training_stats[agent] = [x[:t_min] for x in d]

# Plot discounted returns.
fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for agent, d in eval_stats.items():
    # Compute mean and std.
    mu, std = np.mean(d[:, 2, :], axis=0), np.std(d[:, 2, :], axis=0)
    x = (d[0, 0] / 1e6)

    ax.plot(x, mu, label=agent.upper())
    ax.fill_between(x, mu-std, mu+std, alpha=0.5)

    # Compute best mean
    idx = np.argmax(mu)
    print(f'Best discounted return for {agent}: {mu[idx]} +/- {std[idx]}')

ax.set_xlim([None, 2])
ax.set_xlabel('Timestep (million)')
ax.set_ylabel('Discounted return\ncomputed as in Eqn. (7)')
ax.legend(loc='lower right', fontsize='small')

fig.savefig('figures/discounted_return.pdf', bbox_inches='tight')


# Plot total rewards.
fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for agent, d in eval_stats.items():
    # Compute mean and std.
    mu, std = np.mean(d[:, 1, :], axis=0), np.std(d[:, 1, :], axis=0)
    x = (d[0, 0] / 1e6)

    ax.plot(x, mu, label=agent.upper())
    ax.fill_between(x, mu-std, mu+std, alpha=0.5)

    # Compute best mean
    idx = np.argmax(mu)
    print(f'Best total return for {agent}: {mu[idx]} +/- {std[idx]}')

ax.set_xlim([None, 2])
ax.set_xlabel('Timestep (million)')
ax.set_ylabel('Non-discounted return (T=2000)')
ax.legend(loc='lower right')

fig.savefig('figures/nondiscounted_return.pdf', bbox_inches='tight')

# Plot the evolution of episode length over training.
# fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
#
# for agent, d in training_stats.items():
#     training_lengths = [x['l'] for x in d]
#     mu = np.mean(training_lengths, axis=0)
#     std = np.std(training_lengths, axis=0)
#
#     ax.plot(mu, label=agent.upper())
#     print(f'Longest episode for {agent}: ', np.max(mu))
#
# ax.set_xlabel('Training episode')
# ax.set_ylabel('Episode length')
# ax.legend(loc='lower right')
#
# fig.savefig('figures/training_ep_length.pdf', bbox_inches='tight')


# plt.show()

if __name__ == '__main__':
    print('Done!')