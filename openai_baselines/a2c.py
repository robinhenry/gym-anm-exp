import gym
import argparse
import time

from stable_baselines3 import A2C
from stable_baselines3.a2c.policies import MlpPolicy
from stable_baselines3.common import set_random_seed
from stable_baselines3.common.cmd_util import make_vec_env

from custom_evaluation import EvalCallback


# Read random seed from command line.
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', type=int, default=2020,
                    help='the random seed')
parser.add_argument('-T', '--n_steps', type=int, default=int(1e6),
                    help='number of steps to train the agent for')
args = parser.parse_args()
seed = args.seed
train_timesteps = args.n_steps

# Print arguments.
print('\nArguments\n---------')
for arg, val in sorted(vars(args).items()):
    print('{}: {}'.format(arg, val))
print()

# Non-default hyperparameters
env_id = 'gym_anm:ANM6Easy-v0'
gamma = 0.995

# Separate environments for training and evaluation.
env = make_vec_env(env_id, n_envs=4)
eval_env = gym.make(env_id)

# Set all random seeds.
for e in [env, eval_env]:
    e.seed(seed)
set_random_seed(seed)

# Define a custom callback to periodically evaluate the agent.
n_eval_episodes = 20
eval_freq = int(1e3)
max_steps_per_eval_episode = 3 * int(1e3)
log_path = './results/a2c_' + str(seed) + '/'
eval_callback = EvalCallback(eval_env, best_model_save_path=log_path,
                             log_path=log_path, eval_freq=eval_freq,
                             deterministic=True, render=False,
                             verbose=1,
                             max_steps_per_episode=max_steps_per_eval_episode,
                             gamma=gamma)

# Initialize the model.
model = A2C(MlpPolicy, env, gamma=gamma, verbose=1, seed=seed)

# Train the model.
start = time.time()
model.learn(total_timesteps=train_timesteps, callback=eval_callback)
total_time = time.time() - start

print('Time elapsed to train for %d timesteps: %s.' % (train_timesteps,
                                                       total_time))
