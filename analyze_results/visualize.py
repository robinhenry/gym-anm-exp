"""
This script can be used to visualize a trained-agent interacting
with an environment, assuming the environment supports rendering.

Usage
-----
To visualize a trained agent <ALGO> (SAC or PPO) saved in folder <PATH>,
for <TIMESTEPS> timesteps, pausing for <SLEEP> seconds between each timesteps
(to make it easier to visualize):
    python visualize.py <ALGO> -p <PATH> -s <SLEEP> -T <TIMESTEPS>
"""
import os
import time
import argparse
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO

from rl_agents.hyperparameters import ENV_ID
from rl_agents.utils import load_visualization_env


def visualize(path, algo, T, sleep_time):
    """
    Visualize a trained agent.

    Parameters
    ----------
    path : str
        The path to the folder in which the agent is saved.
    algo : :py:class:`stable_baselines3.sac.SAC` or :py:class:`stable_baselines3.ppo.PPO`
        The class of the trained RL agent.
    T : int
        The number of timesteps to simulate.
    sleep_time : float
        The amount of seconds to sleep between timesteps.
    """

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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", type=str, help="Class of agent to visualize")
    parser.add_argument("--path", '-p', type=str, help='The path to the folder containing the trained agent')
    parser.add_argument("--sleep", '-s', type=float, default=0.5, help='Sleep time between rendering updates')
    parser.add_argument("-T", type=int, default=int(1e4), help='Number of timesteps to render')

    return parser.parse_args()


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
