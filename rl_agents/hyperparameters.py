import os


BASE_DIR = './results/'
ENV_ID = 'gym_anm:ANM6Easy-v0'
GAMMA = 0.995

POLICY = 'MlpPolicy'
TRAIN_STEPS = 3000000
MAX_TRAINING_EP_LENGTH = 5000

EVAL_FREQ = 10000
N_EVAL_EPISODES = 5
MAX_EVAL_EP_LENGTH = 3000

LOG_DIR = BASE_DIR + ENV_ID + '/'
os.makedirs(LOG_DIR, exist_ok=True)

# Create a new directory for this run.
i = 0
while os.path.isdir(LOG_DIR + f'run_{i}/'):
    i += 1
LOG_DIR += f'run_{i}/'
os.makedirs(LOG_DIR, exist_ok=True)

TENSORBOARD_LOG = LOG_DIR + 'tensorboard/'
os.makedirs(TENSORBOARD_LOG, exist_ok=True)
TB_LOG_NAME = 'run'


if __name__ == '__main__':
    print('Done.')
