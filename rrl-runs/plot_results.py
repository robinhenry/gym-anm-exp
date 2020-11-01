import argparse
import pandas as pd

from rrl import agents
from utils import get_result_filename, plot_training_curve, get_plot_filename

from constants import ENVIRONMENTS

# Parse command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('algo', type=str, help='name of the agent algorithm',
                    choices=agents.algos.keys())
parser.add_argument('--seed', type=int, help='random seed',
                    default=2020)
parser.add_argument('--env', type=str, help='name of the environment',
                    default='dipoc', choices=ENVIRONMENTS.keys())
parser.add_argument('-n', '--name', help='name of the training for plots',
                    default='default', type=str)
args = parser.parse_args()


# Plot the training curve.
results_filepath = get_result_filename(args.env, args.algo, args.name,
                                       args.seed)
plot_filename = get_plot_filename(args.env, args.algo, args.name, args.seed)
results = pd.read_csv(results_filepath)
plot_training_curve(results, plot_filename)
