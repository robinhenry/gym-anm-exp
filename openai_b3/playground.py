import torch as th
import numpy as np

from stable_baselines3.common.utils import set_random_seed

from openai_b3.hyperparameters import *
from openai_b3.utils import make_envs


# Overwrite default values.
N_EVAL_EPISODES = 1
EVAL_FREQ = 50
MAX_EVAL_STEPS_PER_EPISODE = 500
TRAIN_STEPS = 1000

# Set random seed
set_random_seed(SEED)

# Environments
train_env, eval_env = make_envs(ENV_ID, LOG_DIR, GAMMA, MAX_EVAL_STEPS_PER_EPISODE, SEED)

# Agent
model = ALGO(POLICY, train_env, verbose=0, tensorboard_log=TENSORBOARD_LOG)

with th.no_grad():
    # Random observation vector (output of VecNormalize during training)
    obs = th.as_tensor(train_env.reset())

    action2, state2 = model.predict(obs, state=None, deterministic=True)

    # Sample action (unclipped)
    actions, values, log_probs = model.policy.forward(obs)
    actions = actions.cpu().numpy()
    clipped_actions = np.clip(actions, model.action_space.low, model.action_space.high)
    new_obs, rewards, dones, infos = train_env.step(clipped_actions)

    print('Done')

if __name__ == '__main__':
    print('Done!')
