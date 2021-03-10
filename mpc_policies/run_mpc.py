"""
This script runs a given MPC policy in a gym-anm environment for a single episode
and repeats for a range of different MPC hyperparameters:
- the number of planning steps (= size of the optimization horizon) N,
- the safety margin hyperparameters \beta (in [0,1]).

The results (non-discounted and discounted returns) are saved in a single .txt file.

For more information about MPC-based policies in `gym-anm`, see the official documentation at
https://gym-anm.readthedocs.io/en/latest/topics/mpc.html.

Examples
--------
The script can be run as follows: ::

    $ python -m mpc_policies.run_mpc <ENV_ID> <POLICY> -T <T> -s <SEED> -o <OUTPUT_FILE>
"""

import gym
from time import time
import argparse

from gym_anm import MPCAgent, MPCAgentANM6Easy


# The hyperparameters to vary between different runs.
PLANNING_STEPS = [8, 16, 32, 64, 128]  # length of the optimization horizon
SAFETY_MARGINS = [0.92, 0.94, 0.96, 0.98, 1.0]  # safety margin :math:`\\beta`` in [0,1]


def grid_search(env_id, policy, T, seed, savefile):
    """
    Run a grid search with the MPC policy and various optimization horizons
    and safety parameters.

    Parameters
    ----------
    env_id : str
        The gym environment ID.
    policy : :py:class:`gym_anm.MPCAgent` or `gym_anm.MPCAgentANM6Easy`
        The MPC policy class.
    T : int
        Total number of timesteps to use to compute the discounted return.
    seed : int
        The random seed.
    savefile : str
        The path to the file in which to write the results.
    """
    print('Using MPC policy: ' + policy.__name__ + f' with T={T}')

    for planning_steps in PLANNING_STEPS:
        for safety_margin in SAFETY_MARGINS:

            run_baseline(env_id, policy, safety_margin, planning_steps,
                         T, seed, savefile)


def run_baseline(env_id, policy, safety_margin, planning_steps, T,
                 seed=None, savefile=None):
    """
    Run an MPC policy for one episode and save its performance.

    Parameters
    ----------
    env_id : str
        The gym environment ID.
    policy : :py:class:`gym_anm.MPCAgent` or `gym_anm.MPCAgentANM6Easy`
        The MPC policy class.
    safety_margin : float
        The safety margin hyperparameter in the MPC formulation :math:`\\beta`` in [0, 1].
    planning_steps : int
        The size of the optimization horizon.
    T : int
        The total number of timesteps to take in the environment to estimate the discounted return.
    seed : int, optional
        The random seed.
    savefile : str
        The path to the file in which to write the results.
    """

    # Get file to write results to.
    if savefile is None:
        savefile = './MPC_{}_results.txt'.format(policy.__name__)

    # Initialize the environment ready.
    env = gym.make(env_id)
    gamma = env.gamma
    if seed is not None:
        env.seed(seed)
    env.reset()

    # Initialize the MPC policy.
    agent = policy(env.simulator, env.action_space, gamma,
                   safety_margin=safety_margin,
                   planning_steps=planning_steps)

    # Run the policy in the environment for T timesteps.
    ret = 0.
    total_reward = 0.
    start = time()
    for i in range(T):
        a = agent.act(env)
        obs, r, done, _ = env.step(a)

        ret += gamma ** i * r
        total_reward += r

        if done:
            env.reset()
            print('Resetting the environment at t=%d.' % i)
    elapsed = time() - start

    # Write results to file.
    with open(savefile, 'a') as f:
        f.write('T=%d, N-stage=%d, safety_margin=%.2f, return=%.3f, total reward=%.3f\n'
                % (T, agent.planning_steps, agent.safety_margin, ret, total_reward))

    print('Planning steps: %d, safety margin: %.2f, return: %.4f, total reward: %.4f, elapsed time: %.2f sec'
          % (agent.planning_steps, agent.safety_margin, ret, total_reward, elapsed))

    return ret


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, help='The environment ID')
    parser.add_argument('policy', type=str, help='Which MPC variant to run')
    parser.add_argument('-T', type=int, help='The number of timesteps to run', default=3000)
    parser.add_argument('--seed', '-s', type=int, help='The random seed', default=None)
    parser.add_argument('--output', '-o', help='The file to write the results to',
                        default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.policy == 'constant':
        policy = MPCAgent
    elif args.policy == 'perfect':
        policy = MPCAgentANM6Easy
    else:
        raise ValueError()

    grid_search(args.env, policy, args.T, args.seed, args.output)
