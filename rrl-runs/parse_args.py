import argparse

from rrl import agents, networks
from constants import ENVIRONMENTS


def parse_arguments(verbose=False):
    """
    Parse the arguments from the stdin
    """
    parser = argparse.ArgumentParser()

    # Seed
    parser.add_argument('--seed', type=int, help='random seed',
                        default=2020)

    # Environment
    parser.add_argument('--env', type=str, help='name of the environment',
                        default='dipoc', choices=ENVIRONMENTS.keys())
    parser.add_argument('--n_envs', type=int,
                        help='the number of environments to run in parallel,'
                             'set to 0 not to use a vectorized environment',
                        default=0)
    parser.add_argument('--masked', type=int, help='state if visible once '
                        'every MASKED frame', default=1)

    # User preferences
    parser.add_argument('-r', '--render', help='render the game during '
                        'training', action='store_true')
    parser.add_argument('--cpu', help='train on CPU', action='store_true')

    # Agent
    parser.add_argument('algo', type=str, help='name of the agent algorithm',
                        choices=agents.algos.keys())
    parser.add_argument('--gamma', help='discount factor', type=float,
                        default=.995)
    parser.add_argument('--rnn', type=str, help='type of rnn', default=None,
                        choices=networks.rnns.keys())
    parser.add_argument('--variant', type=str, help='variant of recurrent '
                        'agent', choices=['rdpg', 'on', 'off'], default='rdpg')

    # Episodes
    parser.add_argument('-M', '--num-episodes', help='number of episodes (if '
                        'terminal states are encountered, there will be more '
                        'episodes such that there is M x T frames in total)',
                        type=int, default=400)
    parser.add_argument('-T', '--num-steps', help='maximum number of '
                        'transitions per episode', type=int, default=1000)
    parser.add_argument('-S', '--eval-period', help='evaluation period in '
                        'number of episodes', type=int, default=10)
    parser.add_argument('-R', '--num_rollouts', help='number of rollouts to '
                        'evaluate performance', type=int, default=20)

    # Outputs
    parser.add_argument('-n', '--name', help='name of the training for plots',
                        default='default', type=str)
    parser.add_argument('-s', '--save', help='save period in number of '
                        'episodes or None for no save', type=int, default=100)
    parser.add_argument('-c', '--continued', help='whether to continue '
                        'training saved earlier (-r, --cpu, --gamma, -H, -L, '
                        '-N, --masked are IGNORED)', action='store_true')

    # RNN parameters
    parser.add_argument('-H', '--rnn-h-size', help='hidden size in each RNN '
                        'layer', type=int, default=256)
    parser.add_argument('-L', '--rnn-layers', help='Number of RNN layers',
                        type=int, default=2)
    parser.add_argument('-N', '--seq-len', help='Length of the sequences used '
                        'to train the RNN', type=int, default=8)
    parser.add_argument('-F', '--ff-sizes', help='Hidden units in the layers '
                        'of the feed-forward layers', type=int, nargs='+',
                        default=(256, 256))

    args = parser.parse_args()

    if verbose:
        print('\nArguments\n---------')
        for arg, val in sorted(vars(args).items()):
            print('{}: {}'.format(arg, val))
        print()

    return args