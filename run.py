# -*- coding: utf-8 -*-
import numpy as np
import time
import gym
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

import agents
from utils.multiprocessing_env import SubprocVecEnv
from parse_args import parse_arguments
from constants import ENVIRONMENTS

sns.set()

VERBOSE = True

def get_result_filename(env, algo, name, seed):
    return f'results/{env}-{algo}-{name}-{seed}.csv'


def get_save_agent_filename(env, algo, name, seed):
    return f'saves/{env}-{algo}-{name}-{seed}.pkl'


def get_plot_filename(env, algo, name, seed):
    return f'plots/{env}-{algo}-{name}-{seed}.png'


def main():
    # Parse arguments
    args = parse_arguments(verbose=VERBOSE)

    # Filenames
    save_filepath = get_save_agent_filename(args.env, args.algo, args.name,
                                            args.seed)
    results_filepath = get_result_filename(args.env, args.algo, args.name,
                                           args.seed)
    plot_filepath = get_plot_filename(args.env, args.algo, args.name, args.seed)

    # Setup of gym and creation of the environment
    gym.logger.set_level(40)
    env, eval_env = make_env(args.env, args.masked, args.render, args.n_envs)

    # Load agent or create new one
    if args.continued:
        if not os.path.exists(save_filepath):
            raise FileNotFoundError('The saved agent has not been found')

        # Statistics of training time, scores, steps and perf are retrieved
        agent, statistics = agents.base.BaseAgent.load(save_filepath, env)
        start_e, total_steps, total_time = statistics

        # On the contrary to other arguments, `render` can be changed
        if args.render:
            agent.render = True

    else:
        Agent = agents.algos[args.algo]
        agent = Agent(env, args.gamma, args.render, cpu=args.cpu, rnn=args.rnn,
                      rnn_layers=args.rnn_layers, rnn_h_size=args.rnn_h_size,
                      seq_len=args.seq_len, ff_sizes=args.ff_sizes,
                      variant=args.variant)

        # Initialize statistics
        start_e, total_steps, total_time = 0, 0, 0.0

    # Prepare outputs
    console_output = 'Episode {:04d} (steps={:d}): {:3.2f} ({:.2f}s)'

    # Create required folders and files.
    os.makedirs('saves', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    if not args.continued or not os.path.exists(results_filepath):
        with open(results_filepath, 'w') as f:
            f.write('env,algo,rnn,n_episodes,n_frames,train_time,')
            f.write('mean_return,std_return\n')

    # Training loop
    episode = start_e
    while total_steps < args.num_steps * args.num_episodes:
        episode += 1

        # Train on one epoch
        start = time.perf_counter()
        cumulative_reward, num_steps = agent.train(args.num_steps)
        elapsed = time.perf_counter() - start

        total_time += elapsed
        total_steps += num_steps

        if VERBOSE:
            print(console_output.format(episode, num_steps, cumulative_reward,
                                        elapsed))

        # Evaluate the current policy R times every S episodes
        if episode % args.eval_period == 0:
            test_score = evaluate(agent, eval_env, args.num_rollouts,
                                  args.num_steps, verbose=VERBOSE)
            mean_score = np.mean(test_score)
            std_score = np.std(test_score)

            # Write statistics
            with open(results_filepath, 'a') as f:
                f.write(f'{args.env},{args.algo}-{args.name},{args.rnn},')
                f.write(f'{episode},{total_steps},{total_time:.4f},')
                f.write(f'{mean_score:.4f},{std_score:.4f}\n')

            # Save the agent
            mean_perf = np.mean(test_score)

            # Update the training curve plot.
            results = pd.read_csv(results_filepath)
            plot_training_curve(results, plot_filepath)

        if args.save is not None and episode % args.save == 0:
            print('Saving into {}...'.format(save_filepath))
            statistics = (episode, total_steps, total_time)
            agent.save(save_filepath, statistics)

    env.close()

    # Print statistics
    time_per_frame = total_time / total_steps
    print('Total training time: {:.2f}'.format(total_time))
    print('Total number of frames: {}'.format(total_steps))
    print('Mean training time per frame: {:.3f}'.format(time_per_frame))


def plot_training_curve(results, filepath):
    """
    Save a plot of the performance evaluated over training.

    Parameters
    ----------
    - results : pd.DataFrame
        The results loaded from the results file.
    - filepath : str
        The file to which to save the plot.
    """

    # Cum return vs. timesteps.
    fig, ax = plt.subplots(figsize=(20, 5))
    x = results['n_frames']
    y = results['mean_return']
    ax.plot(x, y, label=results['algo'][0])
    ax.fill_between(x, y - results['std_return'], y + results['std_return'], alpha=0.4)
    fig.savefig(filepath, bbox_inches='tight')

    plt.close(fig)


def make_env(env_id, masked, render, n_envs):
    """
    Create a (vectorized) environment and a standard environment for evaluation.

    Arguments
    ---------
    - env_id: str
        The environment ID.
    - masked: int
        Period with which the observation is non-zero
    - n_envs: int
        Number of environments to run in parallel. Set to 0 not to use vec env.
    """
    env_name = ENVIRONMENTS[env_id]

    if 'CustomEnv' in env_name:
        tmp = __import__(f'environments', fromlist=(env_name,))
        env_fn = getattr(tmp, env_name)

    else:
        env_fn = lambda: gym.make(env_name)

    # Deal with vectorized environments.
    if n_envs > 0:
        env = [env_fn for i in range(n_envs)]
        env = SubprocVecEnv(env)
    else:
        env = env_fn()

    eval_env = env_fn()

    # In PyBulletEnv, render needs to be called to start rendering engine
    if 'PyBulletEnv' in env_name and render:
        env.render()

    # Corrupt the environement with a mask except every `masked` steps
    # return utils.gym.EnvMask(env, masked)

    return env, eval_env


def evaluate(agent, eval_env, num_rollouts, num_steps, verbose=False):
    """
    Evaluate the performance of the agent on the environment using Monte Carlo
    rollouts.

    Arguments
    ---------
    - agent: class implementing an `eval` method
        Agent trained on the environment and able to take action in it
    - eval_env : gym.Env
        The gym environment on which to evaluate the performance of the agent.
    - num_rollouts: int
        Number of rollout to evaluate the policy performance
    - num_steps_per_env: int
        Time horizon for each rollout
    - verbose: bool
        Whether to print the mean performance of the policy
    """
    performances = []

    for n in range(num_rollouts):
        performance, _ = agent.eval(num_steps, eval_env=eval_env)
        performances.append(performance)

    if verbose:
        mean_scores = np.mean(performances)
        print('Current average performance: {:3.2f}'.format(mean_scores))

    return performances


if __name__ == '__main__':
    main()
