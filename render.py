#!/usr/bin/env python

import argparse
import pickle
import gym
import os

from constants import ENVIRONMENTS
from utils import get_save_agent_filename


# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('env', type=str, help='name of the environment',
                    default='dipoc', choices=ENVIRONMENTS.keys())
parser.add_argument('algo', type=str, help='name of the agent algorithm')
parser.add_argument('name', type=str, help='name given to the agent')
parser.add_argument('seed', type=str, help='random seed of the run', default=2020)
parser.add_argument('delay', type=float, help='sleep time between rendering updates (sec)',
                    default=0)

args = parser.parse_args()


# Load agent
filepath = get_save_agent_filename(args.env, args.algo, args.name, args.seed)

if not os.path.exists(filepath):
    raise FileNotFoundError('The saved agent has not been found')

with open(filepath, 'rb') as f:
    agent = pickle.load(f)


# Create environment
env_infos = ENVIRONMENTS[args.env]
gym.logger.set_level(40)
env = gym.make(env_infos['id'])

# For PyBullet environment, render has to be called before the first trajectory
if 'PyBulletEnv' in env_infos['id']:
    import pybulletgym
    env.render()

agent.env = env
agent.render = True

cumulative_reward, trajectory_length = agent.eval(3000, delay=args.delay)
print(f'Cumulative reward: {cumulative_reward:.2f}')
print(f'Trajectory length: {trajectory_length:.2f}')
