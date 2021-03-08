import os
import time
import argparse
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO

from rl_agents.hyperparameters import ENV_ID

from rl_agents.utils import load_visualization_env


def visualize(path, algo, T, sleep_time):

    # Load agent and environment.
    model = algo.load(os.path.join(path, 'best_model'))
    env = load_visualization_env(ENV_ID, os.path.join(path, 'training_vec_env'), 1)

    # Enjoy trained agent
    obs = env.reset()
    done, state = False, None
    for i in range(T):
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, _, dones, info = env.step(action)
        env.render()
        time.sleep(sleep_time)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", type=str, help="Class of agent to visualize")
    parser.add_argument("--path", '-p', type=str, help='The path to the folder containing the trained agent')
    parser.add_argument("--sleep", '-s', type=float, default=0.5, help='Sleep time between rendering updates')
    parser.add_argument("-T", type=int, default=int(1e4), help='Number of timesteps to render')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    if args.agent == 'PPO':
        ALGO = PPO
    elif args.agent == 'SAC':
        ALGO = SAC
    else:
        raise ValueError('Unimplemented agent ' + args.agent)

    visualize(args.path, ALGO, args.T, args.sleep)

    print('Done.')
