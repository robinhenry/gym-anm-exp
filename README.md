# gym-anm-exp

This repository contains the code used to obtain the experimental results presented in
the paper introducing [`gym-anm`](https://github.com/robinhenry/gym-anm):
```
@misc{henry2021gymanm,
      title={Gym-ANM: Reinforcement Learning Environments for Active Network Management Tasks in Electricity Distribution Systems}, 
      author={Robin Henry and Damien Ernst},
      year={2021},
      eprint={2103.07932},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
which can be accessed [here](https://arxiv.org/abs/2103.07932).

## Code Overview
The code is divided into two folders: 
- `rl_agents/` contains the code used to train RL agents,
- `mpc_policies/` contains the code used to run the Model Predictive Control-based policies
  (for more information, see the [`gym-anm` documentation](https://gym-anm.readthedocs.io/en/latest/topics/mpc.html)).

### Reinforcement Learning Agents
The folder `rl_agents/` mainly contains helper functions, callbacks, and wrappers that
were used alongside the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) RL library:
* `callbacks.py` contains callback functions useful for: 
  * Displaying a progress bar (`ProgressBarCallback`), and
  * Evaluating the agent's current policy (`EvalCallback`). This is a slightly modified version
    of the [original callback](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py#L261)
    that calls `evaluation.py` instead of the original evaluation function.
* `continue_training.py` can be used to continue training an agent using a saved model.
* `evaluation.py` evaluates the agent's current policy. This is a modified version of the 
  [original function](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/evaluation.py)
  which also computes the discounted returns.
* `hyperparameters.py` contains all hyperparameters.
* `train.py` is the main training script.
* `utils.py` contains utility functions to initialize environments, etc.
* `view_progress.py` can be used to display some statistics about the current state of training.
* `wrappers.py` contains wrappers for the environment:
  * `NormalizeActionWrapper` normalizes the action space to lie in `[-1, 1]`.
  * `TimeLimitWrapper` sets a maximum number of steps that can be taken in an environment before needing a reset.

### Model Predictive Control (MPC) Policies
The script `run_mpc.py` can be used to run either the 
[constant forecast policy](https://gym-anm.readthedocs.io/en/latest/topics/mpc.html#constant-forecast) or the
[perfect forecast policy](https://gym-anm.readthedocs.io/en/latest/topics/mpc.html#perfect-forecast) for different
planning steps N (optimization horizon) and safety margin hyperparameters &beta;. 

## Running The Code

### Installation & Requirements
Running the code in this repository requires `Python>=3.8` and the packages listed in `requirements.txt`, which 
can be installed as follows:
```
$ pip install -r requirements.txt
```

### RL Agents
Before starting training the agents, you may want to modify certain hyperparameters in `rl_agents/hyperparameters.py`.
In particular, you should specify the folder in which you want the results to be stored `BASE_DIR`.

#### Training
You can then start training your agent with:
```
$ python -m rl_agents.train <ALGO> -s <SEED>
```
where `<ALGO>` can be either `SAC` or `PPO` and `<SEED>` is an optional random seed.

Results will be saved in a new directory `<BASE_DIR>/<ENV_ID>/run_i/` where `i` is replaced by an integer
so as to create a new directory.

#### Inspecting training status
You can get an overview of the training status of your agents by running:
``` 
$ python -m rl_agents.view_progress
```
which will print some statistics about the results saved in subfolders of `<BASE_DIR>/<ENV_ID>/`.

#### Visualizing the performance of a trained agent
You can watch a trained agent interact with the environment by running:
``` 
$ python -m analyze_results.visualize <ALGO> -p <PATH> -s <SLEEP> -T <TIMESTEPS>
```
where `<PATH>` is the path to the run folder (in the form `<BASE_DIR>/<ENV_ID>/run_<i>/`), `<SLEEP>` is
the amount of seconds between updates of the rendering (default is 0.5), and `<TIMESTEPS>` is the number
of timesteps to run.

#### Recording a video of your trained agent
You can record videos of your trained agent by running:
``` 
$ python -m analyze_results.record_screen <PATH> -l <LENGTH> --fps <FPS>
```
where `<PATH>` is the path to where you want to save the recording, `<LENGTH>` is the duration of the 
recording (seconds) and `<FPS>` is the number of frames/seconds to make.

**NOTE:** The above code will simply record your screen. So you need to have the agent already running 
(see previous section). This feature has not been extensively tested, and similar outcomes can easily
be achieved using tools like [QuickTime Player for Mac](https://libguides.rowan.edu/c.php?g=248114&p=4711659).

### MPC Policies
Either MPC-based policies can be run with the following code:
``` 
$ python -m mpc_policies.run_mpc <ENV_ID> <POLICY> -T <T> -s <SEED> -o <OUTPUT_FILE>
```
where `<POLICY>` can be either `constant` or `perfect`.

The above code will run the policy in the environment `<ENV_ID>` for `<T>` timesteps, repeat
for different safety margins and optimization horizons, and save the final return of each run
in the file `<OUTPUT_FILE>`.

Note that, as specified in [the documentation](https://gym-anm.readthedocs.io/en/latest/topics/mpc.html),
the policy `perfect` will only work in the environment `ANM6Easy-v0`.


## Questions
If you have any questions regarding this implementation, please feel free to contact me at
robin@robinxhenry.com.
