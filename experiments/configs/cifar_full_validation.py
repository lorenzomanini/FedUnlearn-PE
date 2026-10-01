"""20-run CIFAR10 core-score benchmark with a shared online-LiRA bank.

Every original and gold model trains for 40 epochs on the full assigned split.
Each deletion scores a 2,048-record sample of the original and target losses:
this is an explicit efficiency approximation, not full-population curvature.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments import spectral_runner


def build_cases():
    return [
        {
            'subtest': 0,
            'unlearning_method': 'information',
            'unlearning_percentage': float(mass),
            'retrain_epochs': 5,
            'reset_strategy': 'initial',
            'tests': ['LiRA'],
        }
        for mass in np.linspace(0, 100, 10)
    ]


def build_config():
    return {
        'test_name': 'CIFAR_random_full',
        'dataset_name': 'cifar10',
        'num_clients': 10,
        'num_classes': 10,
        'distribution_type': 'random',
        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',
        'trainer_name': 'sgd',
        'train_epochs': 40,
        'learning_rate': 0.01,
        'momentum': 0.9,
        'target_client': 0,
        'num_tests': 20,
        'save_models': False,
        'repetition_seed': 3000,
        'num_shadow_models': 40,
        'lira_seed': 2026,
        'lira_global_variance': True,
        'spectral_rank': 20,
        'spectral_num_power_iters': 5,
        'spectral_max_samples': 2048,
        'spectral_target_max_samples': 2048,
        'spectral_seed': 0,
        'spectral_curvature_backend': 'full',
        'spectral_eigenvalue_min': 1e-6,
        'spectral_eigenvalue_rtol': 1e-5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New job directory under stat_tests/CIFAR')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists; choose a new directory')

    np.random.seed(2026)
    torch.manual_seed(2026)
    spectral_runner.set_batch_sizes(128, 128, 128, 128)
    spectral_runner.run_repeated_tests(
        build_config(), build_cases(), str(args.output),
        num_workers=1, save_models=False, plot=False,
    )


if __name__ == '__main__':
    main()
