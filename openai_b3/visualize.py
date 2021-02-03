import os
from stable_baselines3.sac import SAC
import time

from openai_b3.utils import load_visualization_env


def visualize(path):
    # Load agent and environment.
    model = SAC.load(os.path.join(path, 'best_model'))
    env = load_visualization_env('gym_anm:ANM6Easy-v0', os.path.join(path, 'training_vec_env'), 1)

    # Enjoy trained agent
    obs = env.reset()
    done, state = False, None
    for i in range(1000):
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, _, dones, info = env.step(action)
        rewards = env.get_original_reward()
        env.render()
        time.sleep(0.2)


if __name__ == '__main__':
    # Path to log directory.
    # p = '/Users/uni/Dev/gym-anm_exps/analyse_results/results/gym_anm:ANM6Easy-v0/sac_1/'
    p = '/Volumes/Eluteng/gym/gym_anm:ANM6Easy-v0/run_0/'

    visualize(p)

    print('Done.')
