import matplotlib.pyplot as plt


def get_result_filename(env, algo, name, seed):
    return f'results/{env}-{algo}-{name}-{seed}.csv'


def get_plot_filename(env, algo, name, seed):
    return f'plots/{env}-{algo}-{name}-{seed}.png'


def get_save_agent_filename(env, algo, name, seed):
    return f'saves/{env}-{algo}-{name}-{seed}.pkl'


def plot_training_curve(results, filepath):
    """
    Save a plot of the performance evaluated over training.

    Parameters
    ----------
    - results : pd.DataFrame
        The results loaded from the results file.
    - filepath : str
        The file to which to save the plot.
    """

    # Cum return vs. timesteps.
    fig, ax = plt.subplots(figsize=(20, 5))
    x = results['n_frames']
    y = results['mean_return']
    ax.plot(x, y, label=results['algo'][0])
    ax.fill_between(x, y - results['std_return'], y + results['std_return'], alpha=0.4)
    fig.savefig(filepath, bbox_inches='tight')

    plt.close(fig)


def extract_results(results, env, algos, names):
    """
    Extract results from a .csv that Gaspard uses.

    Parameters
    ----------
    results : pd.DataFrame
        The results corresponding to a wide range of runs.

    Returns
    -------
    dict of {str : pd.DataFrame}
    """

    # Select rows for the specified environment.
    results = results[results['env'] == env]

    dict_of_results = {}

    # Select a specific algorithm or rnn if asked.
    for algo, name in zip(algos, names):
        r = results
        if algo is not None:
            r = r[r['algo'].str.startswith(algo + '-')]
        if name is not None:
            r = r[r['algo'].str.endswith('-' + name)]

        dict_of_results[algo + '-' + name] = r

    return dict_of_results


def plot_multiple_runs(results, filepath):
    """
    Plot on a single plot training curves for different algorithms.

    Parameters
    ----------
    results : dict of {str : pd.DataFrame}
        The training curves of the different runs, indexed by unique names.
    filepath : str
        The paht to the file at which to save the plot.
    """

    # Cum return vs. timesteps.
    fig, ax = plt.subplots(figsize=(20, 5))

    for name, result in results.items():

        x = result['n_frames']
        y = result['mean_return']
        ax.plot(x, y, label=result['algo'][0], legend=name)
        ax.fill_between(x, y - result['std_return'], y + result['std_return'], alpha=0.4)

    fig.savefig(filepath, bbox_inches='tight')