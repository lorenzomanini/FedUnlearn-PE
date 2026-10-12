"""TMLR experiment suites for the curvature core score.

Every suite trains 20 original/gold pairs on one fixed five-client partition, scores
target client 0 once per pair and sweeps the reset score mass. Each case evaluates the
reset model, its recovery and a layer-matched random reset under the same reset value
and recovery scope; one 64-shadow online-LiRA bank audits every case of the suite.

Suites (one Slurm array task each; queue.sh runs them all):
  1. heterogeneity sweep on CIFAR-10, from IID (strength 0) to preferential (1), and
     the arXiv Dirichlet(0.8) split;
  2. the same sweep at three strengths on FashionMNIST;
  3. the diagonal TIS of arXiv 2508.19065, the framework's diagonal case, on matched
     splits and seeds;
  4. estimator sensitivity (rank, solver iterations, curvature samples);
  5. recovery budget (epochs, selected-only versus full-weight recovery);
  6. the core-optimal subspace: top generalized eigenvectors of (H_T, H), with
     damping and rank ablations, and top-curvature scores on the same Gauss-Newton
     curvature to separate the subspace from the curvature type.

Usage:
  python -m experiments.configs.tmlr_paper --list
  python -m experiments.configs.tmlr_paper --download          # once, on a login node
  python -m experiments.configs.tmlr_paper --suite NAME --smoke --device cuda
  python -m experiments.configs.tmlr_paper --suite NAME --output stat_tests/TMLR/run
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments import runner, spectral_runner
from experiments.models import create_resnet
from fisherunlearn import compute_client_information
from fisherunlearn.information.spectral_wip import (
    estimate_core_score, estimate_generalized_core_score,
)


# Fine steps at low mass, where class-specific coordinates are selected first.
MASSES = (2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0)
FULL_WEIGHT_MASSES = (5.0, 10.0, 20.0, 40.0, 60.0, 80.0)
RESET_RULES = ('initial', 'zero')
BUDGET_MASSES = (10.0, 20.0, 40.0)
BUDGET_EPOCHS = (2, 5, 10)
DATASET_CLASSES = {'cifar10': 'CIFAR10', 'FashionMNIST': 'FashionMNIST'}


def case(mass, reset='initial', scope='selected', epochs=1):
    return {
        'subtest': 0,
        'unlearning_method': 'information',
        'unlearning_percentage': float(mass),
        'retrain_epochs': epochs,
        'reset_strategy': reset,
        'recovery_scope': scope,
        'tests': ['LiRA'],
    }


def main_cases():
    """Unchanged original, mass sweeps under both reset values with selected-only and
    full-weight recovery, and a no-reset fine-tuning baseline."""
    cases = [case(0.0)]
    for reset in RESET_RULES:
        cases += [case(mass, reset) for mass in MASSES]
        cases += [case(mass, reset, 'all') for mass in FULL_WEIGHT_MASSES]
    return cases + [case(0.0, 'initial', 'all')]


def ablation_cases():
    return [case(0.0)] + [case(mass, reset) for reset in RESET_RULES for mass in MASSES]


def budget_cases():
    cases = [case(0.0)]
    for scope in ('selected', 'all'):
        cases += [case(mass, 'initial', scope, epochs)
                  for epochs in BUDGET_EPOCHS for mass in BUDGET_MASSES]
    return cases + [case(0.0, 'initial', 'all', epochs) for epochs in BUDGET_EPOCHS]


def base_config(dataset_name):
    return {
        'dataset_name': dataset_name,
        'num_clients': 5,
        'num_classes': 10,
        'model_name': 'resnet18',
        'loss_name': 'cross_entropy',
        'trainer_name': 'sgd',
        'train_epochs': 40,
        'learning_rate': 0.01,
        'momentum': 0.9,
        'target_client': 0,
        'num_tests': 20,
        'repetition_seed': 3000,
        'save_models': 'baseline',
        'num_shadow_models': 64,
        'lira_seed': 2026,
        'lira_global_variance': True,
    }


# Core score: subspace iteration to a 1e-3 relative eigenpair residual (at most 50
# updates) on fixed 2,048-record samples, plus a retained-record control score.
CORE = {
    'spectral_subspace': 'top_curvature',
    'spectral_rank': 20,
    'spectral_num_power_iters': 50,
    'spectral_power_tolerance': 1e-3,
    'spectral_max_samples': 2048,
    'spectral_target_max_samples': 2048,
    'spectral_seed': 0,
    'spectral_curvature_backend': 'full',
    'spectral_eigenvalue_min': 1e-6,
    'spectral_eigenvalue_rtol': 1e-5,
    'spectral_hmp_chunk_size': 1,
    'spectral_control_target': True,
}
# Core-optimal subspace on Gauss-Newton curvature, with H estimated from a retained
# sample plus the target sample itself. Many target-dominated directions sit at the
# bound 1/alpha_T, so the damping (a prior precision, relative to H's top eigenvalue)
# ranks them by curvature; block LOBPCG runs for at most 100 iterations.
GENERALIZED = {
    **CORE,
    'spectral_subspace': 'generalized',
    'spectral_curvature_backend': 'ggn',
    'spectral_num_power_iters': 100,
    'spectral_damping': 1e-4,
}
# Diagonal TIS on the records the model was trained on, without stochastic correction.
DIAGONAL = {
    'hessian_method': 'diag_ggn_mc',
    'stochastic_correction': False,
    'diagonal_population': 'training',
}


def mix(strength):
    return {'distribution_type': 'preferential_mix', 'preferential_strength': strength}


def dirichlet(alpha):
    return {'distribution_type': 'dirichlet', 'dirichlet_alpha': alpha}


SUITES = {}


def add(name, dataset_name, partition, method, cases, **overrides):
    estimator = {'core': CORE, 'generalized': GENERALIZED, 'diagonal': DIAGONAL}[method]
    config = {**base_config(dataset_name), **partition, **estimator, **overrides, 'test_name': name}
    SUITES[name] = {'method': method, 'config': config, 'cases': cases}


# 1. Selectivity should follow the target's curvature signature: strength is the share
#    of the target's exclusive class held by the target (0 is IID, 1 is preferential).
for strength in (1.0, 0.75, 0.5, 0.25, 0.0):
    add(f'c10_mix{round(100 * strength):03d}_core', 'cifar10', mix(strength), 'core', main_cases())
# The partitioner keeps equal client sizes, so this split uses about 57% of the data.
add('c10_dir080_core', 'cifar10', dirichlet(0.8), 'core', main_cases())

# 2. Second dataset.
for strength in (1.0, 0.5, 0.0):
    add(f'fm_mix{round(100 * strength):03d}_core', 'FashionMNIST', mix(strength), 'core', main_cases())

# 3. Diagonal TIS on the same splits, seeds and LiRA candidates.
for strength in (1.0, 0.5, 0.0):
    add(f'c10_mix{round(100 * strength):03d}_diag', 'cifar10', mix(strength), 'diagonal', main_cases())
add('fm_mix100_diag', 'FashionMNIST', mix(1.0), 'diagonal', main_cases())

# 4. Estimator sensitivity on the preferential CIFAR-10 split.
for rank in (10, 40, 80):
    add(f'c10_mix100_rank{rank}', 'cifar10', mix(1.0), 'core', ablation_cases(), spectral_rank=rank)
for iterations in (5, 25):
    add(f'c10_mix100_iters{iterations}', 'cifar10', mix(1.0), 'core', ablation_cases(),
        spectral_num_power_iters=iterations)
for samples in (512, 8192):
    add(f'c10_mix100_samples{samples}', 'cifar10', mix(1.0), 'core', ablation_cases(),
        spectral_max_samples=samples, spectral_target_max_samples=samples)

# 5. Recovery budget.
add('c10_mix100_budget', 'cifar10', mix(1.0), 'core', budget_cases())

# 6. Core-optimal subspace across heterogeneity, then its damping and rank on the IID
#    split, where record-level directions are the only possible target signal.
for strength in (1.0, 0.5, 0.0):
    add(f'c10_mix{round(100 * strength):03d}_gen', 'cifar10', mix(strength), 'generalized', main_cases())
for damping in (1e-6, 1e-2):
    add(f'c10_mix000_gen_damp{damping:.0e}', 'cifar10', mix(0.0), 'generalized', ablation_cases(),
        spectral_damping=damping)
add('c10_mix000_gen_rank40', 'cifar10', mix(0.0), 'generalized', ablation_cases(), spectral_rank=40)
# Top-curvature subspace on the same Gauss-Newton curvature.
for strength in (1.0, 0.0):
    add(f'c10_mix{round(100 * strength):03d}_core_ggn', 'cifar10', mix(strength), 'core', ablation_cases(),
        spectral_curvature_backend='ggn')


def resolve(value):
    names = list(SUITES)
    if str(value).isdigit():
        index = int(value)
        if index >= len(names):
            raise KeyError(f'suite index {index} is outside 0-{len(names) - 1}')
        return names[index]
    if value not in SUITES:
        raise KeyError(f'unknown suite {value!r}; see --list')
    return value


def check_data(dataset_name, download=False):
    """Fail early when a dataset is missing; compute nodes cannot download it."""
    from torchvision import datasets
    dataset_class = getattr(datasets, DATASET_CLASSES[dataset_name])
    for train in (True, False):
        dataset_class(root='./data', train=train, download=download)


def smoke(name, device):
    """Exercise the suite's data and scoring path on the target device without training."""
    suite = SUITES[name]
    config = suite['config']
    check_data(config['dataset_name'])
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable; run the smoke check inside a GPU job')
    torch.manual_seed(2026)
    model = create_resnet(config)
    data = TensorDataset(torch.randn(8, 3, 64, 64), torch.randint(10, (8,)))
    if suite['method'] == 'core':
        loader = DataLoader(data, batch_size=4)
        score = estimate_core_score(
            model.to(device), loader, loader, nn.CrossEntropyLoss(), 2, device, num_power_iters=1,
            curvature_backend=config['spectral_curvature_backend'], loader_control=loader,
        )
        assert 'control' in score['diagnostics']
        values = score['diag_by_name']
    elif suite['method'] == 'generalized':
        retained, target, control = (
            DataLoader(TensorDataset(torch.randn(8, 3, 64, 64), torch.randint(10, (8,))), batch_size=4)
            for _ in range(3))
        score = estimate_generalized_core_score(
            model.to(device), retained, target, nn.CrossEntropyLoss(), 2, device, 0.2,
            num_iters=1, damping=config['spectral_damping'], loader_control=control,
            damping_power_iters=2,
        )
        assert 'control' in score['diagnostics']
        values = score['diag_by_name']
    else:
        runner.set_device(device)
        runner.set_batch_sizes(4, 4, 4, 4)
        values = compute_client_information(
            1, model, nn.CrossEntropyLoss(), [data, data], method=config['hessian_method'],
            use_converter=True, learning_rate=config['learning_rate'], momentum=config['momentum'],
        )
    expected = {key: value for key, value in model.named_parameters() if value.requires_grad}
    assert values.keys() == expected.keys()
    assert all(values[key].shape == expected[key].shape and torch.isfinite(values[key]).all()
               and (values[key] >= 0).all() for key in expected)
    print(f'PASS: {name} ({suite["method"]}) scoring path on {device}')


def run(name, output):
    suite = SUITES[name]
    if (output / name).exists():
        raise SystemExit(f'{output / name} already exists; choose a new output directory')
    # Same seeds before partitioning: core and diagonal suites share splits.
    np.random.seed(2026)
    torch.manual_seed(2026)
    module = runner if suite['method'] == 'diagonal' else spectral_runner
    module.set_batch_sizes(128, 128, 128, 128)
    module.run_repeated_tests(
        dict(suite['config']), [dict(item) for item in suite['cases']], str(output),
        num_workers=1, save_models=suite['config']['save_models'],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list', action='store_true', help='print array index, suite, method and case count')
    parser.add_argument('--name', metavar='INDEX', help='print the suite name for a Slurm array index')
    parser.add_argument('--download', action='store_true', help='download every dataset the suites use')
    parser.add_argument('--suite', help='suite name or array index')
    parser.add_argument('--describe', action='store_true', help='print the suite configuration as JSON')
    parser.add_argument('--smoke', action='store_true', help='check data and the scoring path, then exit')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, help='parent directory; results go to OUTPUT/<suite>')
    args = parser.parse_args()
    if args.list:
        for index, (name, suite) in enumerate(SUITES.items()):
            print(f"{index}\t{name}\t{suite['method']}\t{len(suite['cases'])} cases")
        return
    if args.name is not None:
        print(resolve(args.name))
        return
    if args.download:
        for dataset_name in sorted({suite['config']['dataset_name'] for suite in SUITES.values()}):
            check_data(dataset_name, download=True)
        return
    if args.suite is None:
        parser.error('--suite is required')
    name = resolve(args.suite)
    if args.describe:
        print(json.dumps({'suite': name, **SUITES[name]}, indent=2))
    elif args.smoke:
        smoke(name, args.device)
    elif args.output is None:
        parser.error('--output is required to run a suite')
    else:
        run(name, args.output)


if __name__ == '__main__':
    main()
