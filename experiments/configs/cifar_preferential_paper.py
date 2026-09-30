"""Paper-scale CIFAR-10 preferential-client spectral sweep with online LiRA.

The client partition and 40-model shadow bank are shared across 20 training
seeds. Every one of the ten score masses receives a LiRA audit. Curvature is
estimated from 2,048 records per loss, so this is a sampled-core experiment.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments import spectral_runner


TEST_NAME = 'CIFAR_pref_spectral_paper'
NUM_REPETITIONS = 20
NUM_SHADOW_MODELS = 40
RESET_STRATEGY = 'zero'
RECOVERY_EPOCHS = 1


def build_cases():
    return [
        {
            'subtest': 0,
            'unlearning_method': 'information',
            'unlearning_percentage': float(mass),
            'retrain_epochs': RECOVERY_EPOCHS,
            'reset_strategy': RESET_STRATEGY,
            'tests': ['LiRA'],
        }
        for mass in np.linspace(0, 100, 10)
    ]


def build_config():
    return {
        'test_name': TEST_NAME,
        'dataset_name': 'cifar10',
        'num_clients': 5,
        'num_classes': 10,
        'distribution_type': 'preferential_class',
        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',
        'trainer_name': 'sgd',
        'train_epochs': 40,
        'learning_rate': 0.01,
        'momentum': 0.9,
        'target_client': 0,
        'num_tests': NUM_REPETITIONS,
        'save_models': True,
        'repetition_seed': 3000,
        'num_shadow_models': NUM_SHADOW_MODELS,
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
        num_workers=1, save_models=True, plot=False,
    )


if __name__ == '__main__':
    main()
