"""
This script can be used to continue training a saved agent previously
trained with `train.py`.

It accepts 3 command line arguments: ::

    $ python -m rl_agents.continue_training <ALGO> -p <PATH> -s <SEED>

where
- <ALGO> is the RL algorithm to use in {SAC, PPO},
- <PATH> is a path to the directory in which the agent was
  saved (same as `LOG_DIR` in `hyperparameters.py`),
- <SEED> is a random seed.

All other hyperparameter values are taken from `hyperparameters.py`.
"""
import os
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO

from .hyperparameters import *
from .callbacks import ProgressBarManager, EvalCallback
from .utils import parse_args, make_envs, load_training_env

# Read command line arguments.
args = parse_args()
SEED = args.seed
LOG_DIR = args.path

if args.agent == 'PPO':
    ALGO = PPO
elif args.agent == 'SAC':
    ALGO = SAC
else:
    raise ValueError('Unimplemented agent ' + args.agent)

# Re-create logging folders
TENSORBOARD_LOG = os.path.join(LOG_DIR, 'tensorboard')

# Paths to trained agent and training env
ENV_PATH = os.path.join(LOG_DIR, 'training_vec_env')
AGENT_PATH = os.path.join(LOG_DIR, 'best_model.zip')

# Set random seed
set_random_seed(SEED)

# Load saved training env
train_env = load_training_env(ENV_ID, ENV_PATH, LOG_DIR, MAX_TRAINING_EP_LENGTH, SEED)

# Make eval environment
_, eval_env = make_envs(ENV_ID, LOG_DIR, GAMMA, MAX_TRAINING_EP_LENGTH, MAX_EVAL_EP_LENGTH, SEED)

# Load agent
model = ALGO.load(AGENT_PATH, train_env)

# Callbacks
eval_callback = EvalCallback(eval_env, best_model_save_path=LOG_DIR,
                             log_path=LOG_DIR, eval_freq=EVAL_FREQ,
                             deterministic=True, render=False,
                             n_eval_episodes=N_EVAL_EPISODES, verbose=True)
callbacks = [eval_callback]

# Continue training agent
with ProgressBarManager(TRAIN_STEPS) as c:
    callbacks += [c]
    model.learn(total_timesteps=TRAIN_STEPS, tb_log_name=TB_LOG_NAME, callback=callbacks,
                reset_num_timesteps=False)


if __name__ == '__main__':
    print('Done.')
