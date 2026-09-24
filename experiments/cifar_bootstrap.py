"""Prepare CIFAR models for checkpoint-only recovery ablations.

Fresh mode trains original and gold models on a new saved split. Saved-suite
mode trains one original on an existing split and reuses its gold predictions
and LiRA shadow bank. Neither mode trains shadow models.
"""

import argparse
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

from experiments.datasets import get_datasets
from experiments.evaluation import evaluate_model
from experiments.models import get_model_class
from experiments.training import revised_simple_trainer


def read_pickle(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


def write_pickle(path, value):
    with path.open('wb') as stream:
        pickle.dump(value, stream)


def bootstrap(source_suite, repetition, output, seed=2026, batch_size=128):
    """Return paths of a fresh original checkpoint and its matching suite."""
    overall_start = time.perf_counter()
    source_suite, output = Path(source_suite), Path(output)
    if output.exists():
        raise FileExistsError(f'Bootstrap output already exists: {output}')
    source_iteration = source_suite / f'test_{repetition}'
    config = read_pickle(source_suite / 'init_params.pkl')
    if config['dataset_name'] != 'cifar10' or config['model_name'] != 'resnet18':
        raise ValueError('Bootstrap expects the saved CIFAR10/ResNet18 suite')
    saved = read_pickle(source_iteration / 'score_diagnostics.pkl')
    labels = read_pickle(source_suite / 'labels.pkl')
    train_ids, target_ids = list(saved['training_indices']), list(saved['target_indices'])
    train_data, test_data = get_datasets(config)
    if len(train_ids) != len(set(train_ids)) or len(target_ids) != len(set(target_ids)):
        raise ValueError('Saved training/target indices must be unique')
    if not set(target_ids).issubset(train_ids):
        raise ValueError('Saved target records must be in the original training set')
    if not np.array_equal(np.asarray(train_data.targets), np.asarray(labels['train'])):
        raise ValueError('Downloaded CIFAR training labels do not match saved labels')
    if not np.array_equal(np.asarray(test_data.targets), np.asarray(labels['test'])):
        raise ValueError('Downloaded CIFAR test labels do not match saved labels')

    validation_ids = sorted(set(range(len(train_data))) - set(train_ids))
    bank_file = source_suite / 'lira_shadow_bank.npz'
    if bank_file.exists():
        with np.load(bank_file) as bank:
            candidates = set(bank['candidate_complete_indices'].tolist())
        validation_ids = [i for i in validation_ids if i not in candidates]
    if not validation_ids:
        raise ValueError('Saved split has no retained validation records')

    torch.manual_seed(seed)
    np.random.seed(seed)
    model_class = get_model_class(config)
    started = time.perf_counter()
    trained = revised_simple_trainer(
        model_class(), nn.CrossEntropyLoss(),
        [Subset(train_data, train_ids)], [Subset(train_data, validation_ids)],
        config['train_epochs'], config, train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )
    training_seconds = time.perf_counter() - started
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_train = evaluate_model(trained, train_data, device, batch_size)
    new_test = evaluate_model(trained, test_data, device, batch_size)
    previous_train = read_pickle(source_iteration / 'initial_eval_train_results.pkl')
    previous_test = read_pickle(source_iteration / 'initial_eval_test_results.pkl')
    stage = read_pickle(source_iteration / 'stage_timings.pkl')
    if len(previous_train['shadow_out']['pred']) != len(train_data):
        raise ValueError('Saved gold training predictions do not match CIFAR train size')
    if len(previous_test['shadow_out']['pred']) != len(test_data):
        raise ValueError('Saved gold test predictions do not match CIFAR test size')

    iteration = output / 'suite' / f'test_{repetition}'
    iteration.mkdir(parents=True, exist_ok=False)
    suite = iteration.parent
    checkpoint = output / 'original_model.pth'
    torch.save(trained.state_dict(), checkpoint)
    write_pickle(suite / 'init_params.pkl', config)
    shutil.copy2(source_suite / 'labels.pkl', suite / 'labels.pkl')
    if bank_file.exists():
        shutil.copy2(bank_file, suite / 'lira_shadow_bank.npz')
    # The ablation reads these indices; old eigenvalues/scores are intentionally
    # omitted because they were computed for a different original model.
    write_pickle(iteration / 'score_diagnostics.pkl', {
        'training_indices': train_ids, 'target_indices': target_ids,
    })
    write_pickle(iteration / 'stage_timings.pkl', {
        'gold_retraining_seconds': stage['gold_retraining_seconds'],
        'bootstrap_original_training_seconds': training_seconds,
    })
    write_pickle(iteration / 'initial_eval_train_results.pkl', {
        'trained': new_train, 'shadow_out': previous_train['shadow_out'],
    })
    write_pickle(iteration / 'initial_eval_test_results.pkl', {
        'trained': new_test, 'shadow_out': previous_test['shadow_out'],
    })
    manifest = {
        'source_suite': str(source_suite), 'repetition': repetition,
        'seed': seed, 'batch_size': batch_size,
        'checkpoint': str(checkpoint), 'suite': str(suite),
        'original_training_seconds': training_seconds,
        'total_bootstrap_seconds': time.perf_counter() - overall_start,
        'gold_retraining_seconds_from_previous_run': stage['gold_retraining_seconds'],
        'note': 'Fresh original model on saved split; previous gold predictions and LiRA bank reused. This is a new experiment, not a reproduction of the old original weights.',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return checkpoint, suite


def bootstrap_fresh(output, seed=2026, batch_size=128, train_epochs=40):
    """Build a complete utility-comparison suite when no saved artifacts exist."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Bootstrap output already exists: {output}')
    overall_start = time.perf_counter()
    config = {
        'test_name': 'CIFAR_random_fresh', 'dataset_name': 'cifar10',
        'model_name': 'resnet18', 'num_classes': 10, 'num_clients': 10,
        'distribution_type': 'random', 'target_client': 0,
        'loss_name': 'cross_entropy', 'trainer_name': 'sgd',
        'learning_rate': 0.01, 'momentum': 0.9,
        'train_epochs': train_epochs, 'spectral_rank': 10,
        'spectral_num_power_iters': 3, 'spectral_max_samples': 512,
        'spectral_target_max_samples': 512, 'spectral_seed': 0,
        'spectral_eigenvalue_min': 1e-6, 'spectral_eigenvalue_rtol': 1e-5,
        'num_tests': 1, 'lira_enabled': False,
    }
    train_data, test_data = get_datasets(config)
    if len(train_data) < 20:
        raise ValueError('CIFAR bootstrap needs enough records for target, training and validation')
    order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(seed)).tolist()
    target_size = len(order) // 10
    target_ids = order[:target_size]
    retained = order[target_size:]
    heldout_size = len(retained) // 10
    heldout_ids = retained[:heldout_size]
    retained_train = retained[heldout_size:]
    train_ids = retained_train + target_ids
    validation = [Subset(train_data, heldout_ids)]
    model_class = get_model_class(config)

    torch.manual_seed(seed + 1)
    original_start = time.perf_counter()
    original = revised_simple_trainer(
        model_class(), nn.CrossEntropyLoss(), [Subset(train_data, train_ids)],
        validation, train_epochs, config, train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )
    original_seconds = time.perf_counter() - original_start

    torch.manual_seed(seed + 2)
    gold_start = time.perf_counter()
    gold = revised_simple_trainer(
        model_class(), nn.CrossEntropyLoss(), [Subset(train_data, retained_train)],
        validation, train_epochs, config, train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )
    gold_seconds = time.perf_counter() - gold_start

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initial_train = {
        'trained': evaluate_model(original, train_data, device, batch_size),
        'shadow_out': evaluate_model(gold, train_data, device, batch_size),
    }
    initial_test = {
        'trained': evaluate_model(original, test_data, device, batch_size),
        'shadow_out': evaluate_model(gold, test_data, device, batch_size),
    }
    iteration = output / 'suite' / 'test_0'
    iteration.mkdir(parents=True, exist_ok=False)
    suite = iteration.parent
    checkpoint = output / 'original_model.pth'
    gold_checkpoint = output / 'gold_model.pth'
    torch.save(original.state_dict(), checkpoint)
    torch.save(gold.state_dict(), gold_checkpoint)
    write_pickle(suite / 'init_params.pkl', config)
    write_pickle(suite / 'labels.pkl', {
        'train': np.asarray(train_data.targets),
        'test': np.asarray(test_data.targets),
    })
    write_pickle(iteration / 'score_diagnostics.pkl', {
        'training_indices': train_ids, 'target_indices': target_ids,
    })
    write_pickle(iteration / 'stage_timings.pkl', {
        'initial_training_seconds': original_seconds,
        'gold_retraining_seconds': gold_seconds,
    })
    write_pickle(iteration / 'initial_eval_train_results.pkl', initial_train)
    write_pickle(iteration / 'initial_eval_test_results.pkl', initial_test)
    manifest = {
        'mode': 'fresh', 'seed': seed, 'batch_size': batch_size,
        'checkpoint': str(checkpoint), 'gold_checkpoint': str(gold_checkpoint),
        'suite': str(suite),
        'train_records': len(train_ids), 'retained_train_records': len(retained_train),
        'target_records': len(target_ids), 'heldout_records': len(heldout_ids),
        'original_training_seconds': original_seconds,
        'gold_retraining_seconds': gold_seconds,
        'total_bootstrap_seconds': time.perf_counter() - overall_start,
        'privacy_status': 'No LiRA shadow bank; audit forgetting and MIA separately after selecting a utility-feasible candidate.',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return checkpoint, suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-suite', type=Path,
                        help='Existing suite; omit when starting from CIFAR data alone')
    parser.add_argument('--repetition', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--train-epochs', type=int, default=40)
    args = parser.parse_args()
    if args.repetition < 0 or args.batch_size < 1 or args.train_epochs < 1:
        parser.error('Repetition must be nonnegative; batch size and epochs positive')
    if args.source_suite is None:
        if args.repetition != 0:
            parser.error('Fresh mode creates repetition 0; omit --repetition')
        checkpoint, suite = bootstrap_fresh(
            args.output, seed=args.seed, batch_size=args.batch_size,
            train_epochs=args.train_epochs,
        )
    else:
        checkpoint, suite = bootstrap(
            args.source_suite, args.repetition, args.output,
            seed=args.seed, batch_size=args.batch_size,
        )
    print(f'Original checkpoint: {checkpoint}\nMatching suite: {suite}', flush=True)


if __name__ == '__main__':
    main()
