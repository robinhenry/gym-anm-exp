import matplotlib.pyplot as plt


def set_rc_params():
    # Plot parameters.
    plt.rcdefaults()

    # Lines
    lines = {'linewidth': 1.5,
             'markersize': 10}
    plt.rc('lines', **lines)

    # Axes.
    axes = {'xmargin': 0.02,
            'labelpad': 6}
    plt.rc('axes', **axes)

    # Fonts.
    font = {'family': 'serif',
            'size': 15}
    plt.rc('font', **font)

    # Legend.
    legend = {'fontsize': '10',
              'loc': 'best'}
    plt.rc('legend', **legend)

    # PQ_org-axis ticks.
    xtick = {'labelsize': 13,
             'major.size': 7}
    plt.rc('xtick', **xtick)

    # Y-axis ticks
    ytick = {'labelsize': 13,
             'major.size': 7,
             'minor.size': 4}
    plt.rc('ytick', **ytick)

    return