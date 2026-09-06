"""Compatibility entry point for revised experiment result analysis."""

from analysis.results import *


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot online-LiRA experiment results.")
    parser.add_argument("test_path", help="Experiment suite directory")
    parser.add_argument("--num-tests", type=int)
    parser.add_argument("--target-fpr", type=float, default=0.001)
    arguments = parser.parse_args()
    plot_experiment_results(
        arguments.test_path,
        num_tests=arguments.num_tests,
        target_fpr=arguments.target_fpr,
    )
