"""
This script can be used to train PPO or SAC RL agents using
the Stable Baselines3 implementations.

It accepts 2 command line arguments: ::

    $ python -m rl_agents.continue_training <ALGO> -s <SEED>

where
- <ALGO> is the RL algorithm to use in {SAC, PPO},
- <SEED> is a random seed.

All other hyperparameter values are taken from `hyperparameters.py`.
"""
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO

from .hyperparameters import *
from .callbacks import ProgressBarManager, EvalCallback
from .utils import make_envs, save_hyperparameters, parse_args, make_log_dirs


# Read command line arguments.
args = parse_args()
SEED = args.seed

if args.agent == 'PPO':
    ALGO = PPO
elif args.agent == 'SAC':
    ALGO = SAC
else:
    raise ValueError('Unimplemented agent ' + args.agent)

# Make log directories.
LOG_DIR, TENSORBOARD_LOG = make_log_dirs(LOG_DIR)

# Write parameters to .txt file in log directory.
PARAMS = {
    'SEED': SEED,
    'ENV_ID': ENV_ID,
    'LOG_DIR': LOG_DIR,
    'MAX_TRAINING_EP_LENGTH': MAX_TRAINING_EP_LENGTH,
    'EVAL_FREQ': EVAL_FREQ,
    'N_EVAL_EPISODES': N_EVAL_EPISODES,
    'TENSORBOARD_LOG': TENSORBOARD_LOG,
    'TB_LOG_NAME': TB_LOG_NAME,
    'POLICY': POLICY,
    'ALGO': ALGO.__name__,
    'TRAIN_STEPS': TRAIN_STEPS,
    'GAMMA': GAMMA,
    'MAX_EVAL_EP_LENGTH': MAX_EVAL_EP_LENGTH,
}
save_hyperparameters(LOG_DIR, PARAMS)

# Set random seed
set_random_seed(SEED)

# Environments
train_env, eval_env = make_envs(ENV_ID, LOG_DIR, GAMMA, MAX_TRAINING_EP_LENGTH, MAX_EVAL_EP_LENGTH, SEED)

# Callbacks
eval_callback = EvalCallback(eval_env, best_model_save_path=LOG_DIR,
                             log_path=LOG_DIR, eval_freq=EVAL_FREQ,
                             deterministic=True, render=False,
                             n_eval_episodes=N_EVAL_EPISODES, verbose=True)
callbacks = [eval_callback]

# Agent
model = ALGO(POLICY, train_env, gamma=GAMMA, verbose=0, tensorboard_log=TENSORBOARD_LOG)

# Train agent
with ProgressBarManager(TRAIN_STEPS) as c:
    callbacks += [c]
    model.learn(total_timesteps=TRAIN_STEPS, tb_log_name=TB_LOG_NAME, callback=callbacks)


if __name__ == '__main__':
    print('Done!')
