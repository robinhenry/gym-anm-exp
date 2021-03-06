from gym_anm import MPCAgentANM6Easy
from baselines.run_baseline import run_baseline


def mpc_grid_search():
    T = 3000
    seed = 1000
    savefile = f'./MPC_returns_T{T}_new_obj.txt'

    for planning_steps in [8, 16, 32, 64, 128]:
        for safety_margin in [0.92, 0.94, 0.96, 0.98, 1.]:

            run_baseline(MPCAgentANM6Easy, safety_margin, planning_steps, T,
                         seed, savefile)


if __name__ == '__main__':
    # mpc_grid_search()

    run_baseline(MPCAgentANM6Easy, 0.96, 64, 3000, 1, '_.txt')
