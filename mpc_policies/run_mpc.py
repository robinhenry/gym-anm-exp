import gym
from time import time
import argparse

from gym_anm import MPCAgent, MPCAgentANM6Easy


def grid_search(env_id, policy, T, seed, savefile):
    for planning_steps in [8, 16, 32, 64, 128]:
        for safety_margin in [0.92, 0.94, 0.96, 0.98, 1.0]:

            run_baseline(env_id, policy, safety_margin, planning_steps,
                         T, seed, savefile)


def run_baseline(env_id, agent_class, safety_margin, planning_steps, T,
                 seed=None, savefile=None):

    print('Using agent: ' + agent_class.__name__ + f' with T={T}')

    # Get file to write results to.
    if savefile is None:
        savefile = './Baseline_{}_results.txt'.format(agent_class.__name__)

    # Get the environment ready.
    env = gym.make(env_id)
    gamma = env.gamma
    if seed is not None:
        env.seed(1000)
    env.reset()
    ret = 0.
    total_reward = 0.

    # Make the agent.
    agent = agent_class(env.simulator, env.action_space, gamma,
                        safety_margin=safety_margin,
                        planning_steps=planning_steps)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, help='The environment ID')
    parser.add_argument('policy', type=str, help='Which MPC variant to run')
    parser.add_argument('-T', type=int, help='The number of timesteps to run', default=3000)
    parser.add_argument('--seed', '-s', help='The random seed', default=None)
    parser.add_argument('--output', '-o', help='The file to write the results to',
                        default=None)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # savefile = f'./DCOPF_returns_T{T}_new_obj.txt'
    # seed = 1000

    args = parse_args()

    if args.policy == 'constant':
        policy = MPCAgent
    elif args.policy == 'perfect':
        policy = MPCAgentANM6Easy
    else:
        raise ValueError()

    grid_search(args.env, policy, args.T, args.seed, args.output)
