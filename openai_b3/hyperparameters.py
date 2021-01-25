import os


SEED = 1

ENV_ID = 'gym_anm:ANM6Easy-v0'
GAMMA = 0.995

POLICY = 'MlpPolicy'
TRAIN_STEPS = 10000000

EVAL_FREQ = 5000
N_EVAL_EPISODES = 1
MAX_EVAL_STEPS_PER_EPISODE = 2000

BASE_DIR = '/Volumes/Eluteng/gym/'
LOG_DIR = BASE_DIR + ENV_ID + '/'
os.makedirs(LOG_DIR, exist_ok=True)

# Create a new directory for this run.
i = 0
while os.path.isdir(LOG_DIR + f'run_{i}/'):
    i += 1
LOG_DIR += f'run_{i}/'
os.makedirs(LOG_DIR)

TENSORBOARD_LOG = LOG_DIR + 'tensorboard'
os.makedirs(TENSORBOARD_LOG, exist_ok=True)
TB_LOG_NAME = 'run'


if __name__ == '__main__':
    print('Done.')
