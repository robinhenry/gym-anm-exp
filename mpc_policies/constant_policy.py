from gym_anm import DCOPFAgent
from baselines.run_baseline import run_baseline


def dcopf_grid_search():
    T = 3000
    seed = 1000
    savefile = f'./DCOPF_returns_T{T}_new_obj.txt'

    for planning_steps in [8, 16, 32, 64]:
        for safety_margin in [0.92, 0.94, 0.96, 0.98, 1.0]:

            run_baseline(DCOPFAgent, safety_margin, planning_steps, T,
                         seed, savefile)


if __name__ == '__main__':
    dcopf_grid_search()
