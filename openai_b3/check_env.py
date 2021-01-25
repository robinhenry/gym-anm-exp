import gym

from stable_baselines3.common.env_checker import check_env

env = gym.make('gym_anm:ANM6Easy-v0')
check_env(env)

if __name__ == '__main__':
    print('Done!')
