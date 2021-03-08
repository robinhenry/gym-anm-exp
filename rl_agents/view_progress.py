import os
import numpy as np

from .hyperparameters import BASE_DIR, ENV_ID


FOLDER = os.path.join(BASE_DIR, ENV_ID)
run_folders = os.listdir(FOLDER)
print(run_folders)

for run_folder in run_folders:
    if run_folder[:3] != 'run':
        continue
    run_id = run_folder.split('_')[-1]

    # Load last evaluation results.
    p = os.path.join(FOLDER, run_folder, 'evaluations.npz')
    data = np.load(p)
    timestep = data['timesteps'][-1]
    result = data['results'][-1, 0, 0]
    disc_result = data['discounted_results'][-1, 0, 0]
    ep_length = data['ep_lengths'][-1, 0]

    # Load elapsed time from `monitor.csv`
    p = os.path.join(FOLDER, run_folder, 'monitor.csv')
    with open(p, 'r') as f:
        r, l, t = f.readlines()[-1].split(',')

    # Load name of algo from `hyperparameters.txt`.
    p = os.path.join(FOLDER, run_folder, 'hyperparameters.txt')
    with open(p, 'r') as f:
        for line in f.readlines():
            if 'ALGO' in line:
                algo = line.split(':')[-1][:-1]

    print(f'RUN_ID: {run_id}, ALGO: {algo}, timestep={timestep}, return={result:.1f}, disc_return={disc_result:.1f}, '
          f'ep_length={ep_length}, t={float(t)/3600:.1f} hours')


if __name__ == '__main__':
    print('Done.')
