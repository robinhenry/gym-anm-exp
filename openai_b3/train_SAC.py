from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.sac import SAC

from openai_b3.hyperparameters import *
from openai_b3.callbacks import ProgressBarManager, EvalCallback
from openai_b3.utils import make_envs, save_hyperparameters


# Overwrite hyperparameters.
ALGO = SAC

# Write parameters to .txt file in log directory.
PARAMS = {
    'ENV_ID': ENV_ID,
    'LOG_DIR': LOG_DIR,
    'EVAL_FREQ': EVAL_FREQ,
    'N_EVAL_EPISODES': N_EVAL_EPISODES,
    'POLICY': POLICY,
    'ALGO': ALGO.__name__,
    'TRAIN_STEPS': TRAIN_STEPS,
    'GAMMA': GAMMA,
    'MAX_EVAL_STEPS_PER_EPISODE ': MAX_EVAL_STEPS_PER_EPISODE,
}
save_hyperparameters(LOG_DIR, PARAMS)

# Set random seed
set_random_seed(SEED)

# Environments
train_env, eval_env = make_envs(ENV_ID, LOG_DIR, GAMMA, MAX_EVAL_STEPS_PER_EPISODE, SEED)

# Callbacks
eval_callback = EvalCallback(eval_env, best_model_save_path=LOG_DIR,
                             log_path=LOG_DIR, eval_freq=EVAL_FREQ,
                             deterministic=True, render=False,
                             n_eval_episodes=N_EVAL_EPISODES, verbose=True)
callbacks = [eval_callback]

# Agent
model = ALGO(POLICY, train_env, verbose=0, tensorboard_log=TENSORBOARD_LOG)

# Train agent
with ProgressBarManager(TRAIN_STEPS) as c:
    callbacks += [c]
    model.learn(total_timesteps=TRAIN_STEPS, tb_log_name=TB_LOG_NAME, callback=callbacks)

# Continue training.
# model.learn(total_timesteps=10000, tb_log_name='second_run', reset_num_timesteps=false)


if __name__ == '__main__':
    print('Done!')
