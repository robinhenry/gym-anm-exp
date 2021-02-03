import gym
import argparse
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from openai_b3.wrappers import NormalizeActionWrapper, TimeLimitWrapper


def make_envs(env_id, log_dir, gamma, max_train_ep_length, max_eval_ep_length, seed):

    # Training env
    train_env = gym.make(env_id)
    train_env.seed(seed)                                          # Set random seed
    train_env = TimeLimitWrapper(train_env, max_train_ep_length)  # Limit length of training episodes
    train_env = Monitor(train_env, log_dir)                       # Monitor training
    train_env = NormalizeActionWrapper(train_env)                 # Normalize action space
    train_env = DummyVecEnv([lambda: train_env])                  # Vectorize environment
    train_env = VecNormalize(train_env, gamma=gamma)              # Normalise observations and rewards

    # Eval env
    eval_env = gym.make(env_id)
    eval_env.seed(seed)                                           # Set random seed
    eval_env = TimeLimitWrapper(eval_env, max_eval_ep_length)     # Set a maximum number of timesteps during eval
    eval_env = Monitor(eval_env)  # Used to ensure original action space is not modified by `NormalizeActionWrapper`
    eval_env = NormalizeActionWrapper(eval_env)                   # Normalize action space
    eval_env = DummyVecEnv([lambda: eval_env])                    # Vectorize environment
    eval_env = VecNormalize(eval_env, gamma=gamma, training=False, norm_reward=False)  # Normalise observations
    # (obs/reward normalization gets synchronised with `train_env` in `EvalCallback`)

    return train_env, eval_env


def load_visualization_env(env_id, env_path, seed=0):
    env = gym.make(env_id)
    env.seed(seed)
    env = Monitor(env)  # Used to ensure original action space is not modified by `NormalizeActionWrapper`
    env = NormalizeActionWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(env_path, env)

    return env


def save_hyperparameters(log_dir, params):
    with open(log_dir + 'hyperparameters.txt', 'w') as f:
        for k in params.keys():
            f.write("'{}':'{}'\n".format(k, params[k]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", type=str, help="Agent to train")
    parser.add_argument("--seed", '-s', type=int, help='Random seed', default=1)

    args = parser.parse_args()

    return args
