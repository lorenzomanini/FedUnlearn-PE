"""Checkpoint-only CIFAR ablation; never trains originals, gold models or shadows.

Run with ``python -m experiments.cifar_recovery_ablation --help``.
Budget checks occur between phases/batches; an in-flight curvature call can
overrun the budget, in which case recovery is refused and the run is failed.
Final reporting and serialization are timed separately from the deletion.
"""

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from experiments.datasets import get_datasets
from experiments.evaluation import evaluate_lira, evaluate_model
from experiments.models import get_model_class
from experiments.runner import _curvature_loader
from fisherunlearn.information.selection import find_informative_params
from fisherunlearn.information.spectral_wip import estimate_core_score
from fisherunlearn.unlearning import UnlearnNet


def read_pickle(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


def synchronized_time(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def recover(model, train_loader, validation_loader, *, epochs, learning_rate,
            momentum, device, deadline, batchnorm_mode='train'):
    """Keep optimizer momentum across epochs; validation never uses targets."""
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    history = []

    def check_budget():
        if synchronized_time(device) >= deadline:
            raise TimeoutError('Deletion budget exhausted; candidate is not validated.')

    for epoch in range(epochs):
        check_budget()
        model.train()
        if batchnorm_mode == 'frozen':
            for module in model.inner_model['model'].modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        total_loss, total = 0.0, 0
        for x, y in train_loader:
            check_budget()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total += len(y)
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in validation_loader:
                check_budget()
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += nn.functional.cross_entropy(logits, y, reduction='sum').item()
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += len(y)
        check_budget()
        history.append(dict(epoch=epoch + 1, train_cross_entropy=total_loss / total,
                            validation_cross_entropy=val_loss / val_total,
                            validation_accuracy=val_correct / val_total))
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=Path('stat_tests/CIFAR'))
    parser.add_argument('--repetition', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Original trained model state_dict for this repetition')
    parser.add_argument('--output', type=Path, required=True,
                        help='New output directory (must not already exist)')
    parser.add_argument('--reset-strategy', choices=['zero', 'initial'], default='initial')
    parser.add_argument('--batchnorm-mode', choices=['train', 'frozen'], default='train')
    parser.add_argument('--score-mass', type=float, default=100 * 4 / 9)
    parser.add_argument('--epochs', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--max-samples', type=int, default=512)
    parser.add_argument('--target-max-samples', type=int, default=512)
    parser.add_argument('--power-iters', type=int, default=3)
    parser.add_argument('--rank', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--budget-fraction', type=float, default=0.25,
                        help='Deletion budget / saved gold retraining time, strictly below 1')
    args = parser.parse_args()
    if not 0 < args.budget_fraction < 1 or not math.isfinite(args.budget_fraction):
        parser.error('--budget-fraction must be strictly between 0 and 1')
    if not 0 < args.score_mass <= 100 or not math.isfinite(args.score_mass):
        parser.error('--score-mass must be in (0,100]')
    if min(args.max_samples, args.target_max_samples, args.rank, args.batch_size) < 1 or args.power_iters < 0:
        parser.error('Sample caps, rank and batch size must be positive; iterations nonnegative')
    if not args.checkpoint.is_file():
        parser.error('--checkpoint must name an existing original-model state_dict')
    if args.output.exists():
        parser.error('--output already exists; choose a new directory')

    config = read_pickle(args.suite / 'init_params.pkl')
    if config['dataset_name'] != 'cifar10' or config['model_name'] != 'resnet18':
        parser.error('This bounded ablation supports the supplied CIFAR10/ResNet18 suite')
    folder = args.suite / f'test_{args.repetition}'
    saved = read_pickle(folder / 'score_diagnostics.pkl')
    labels = read_pickle(args.suite / 'labels.pkl')
    gold_seconds = read_pickle(folder / 'stage_timings.pkl')['gold_retraining_seconds']
    train_data, test_data = get_datasets(config)
    train_ids, target_ids = saved['training_indices'], saved['target_indices']
    target_set = set(target_ids)
    retained_ids = [i for i in train_ids if i not in target_set]
    validation_ids = sorted(set(range(len(train_data))) - set(train_ids))
    bank_path = args.suite / 'lira_shadow_bank.npz'
    bank = dict(np.load(bank_path)) if bank_path.exists() else None
    if bank is not None:
        candidates = set(bank['candidate_complete_indices'].tolist())
        validation_ids = [i for i in validation_ids if i not in candidates]
    if not retained_ids or not validation_ids or not target_set.issubset(train_ids):
        raise ValueError('Invalid or empty saved training/validation split')
    model_class = get_model_class(config)
    model = model_class().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check correspondence before spending a deletion budget on the wrong model.
    # Dataset setup and this provenance check are ablation setup, reported apart.
    setup_start = synchronized_time(device)
    check_ids = train_ids[:128]
    check = evaluate_model(model, Subset(train_data, check_ids), device, args.batch_size)
    original = read_pickle(folder / 'initial_eval_train_results.pkl')['trained']
    if not np.array_equal(check['pred'], original['pred'][check_ids]) or not np.allclose(
        check['loss'], original['loss'][check_ids], rtol=1e-3, atol=1e-3,
    ):
        raise ValueError('Checkpoint does not match saved predictions/margins on 128 records')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'gold_retraining_seconds': gold_seconds,
              'budget_seconds': gold_seconds * args.budget_fraction,
              'provenance_check_seconds': synchronized_time(device) - setup_start,
              'status': 'running'}
    start = synchronized_time(device)
    deadline = start + report['budget_seconds']
    try:
        model.to(device)
        torch.manual_seed(args.seed)
        score = estimate_core_score(
            model, _curvature_loader(Subset(train_data, train_ids), args.max_samples, args.batch_size, args.seed),
            _curvature_loader(Subset(train_data, target_ids), args.target_max_samples, args.batch_size, args.seed + 1),
            nn.CrossEntropyLoss(), args.rank, device, num_power_iters=args.power_iters,
            eigenvalue_threshold=config.get('spectral_eigenvalue_min', 1e-6),
            relative_eigenvalue_threshold=config.get('spectral_eigenvalue_rtol', 1e-5),
            power_tolerance=config.get('spectral_power_tolerance', 1e-3),
        )
        report['score_seconds'] = synchronized_time(device) - start
        report['score_diagnostics'] = score['diagnostics']
        if synchronized_time(device) >= deadline:
            raise TimeoutError('Score exceeded deletion budget; recovery was not started')
        selected = find_informative_params(score['diag_by_name'], 'information', args.score_mass)
        report['selected_per_tensor'] = {k: len(v) for k, v in selected.items()}
        report['selected_parameters'] = sum(report['selected_per_tensor'].values())
        report['batchnorm_scale_bias_overlap'] = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.affine:
                scale = selected.get(name + '.weight', torch.empty(0, 1, dtype=torch.long))
                bias = selected.get(name + '.bias', torch.empty(0, 1, dtype=torch.long))
                report['batchnorm_scale_bias_overlap'][name] = len(
                    set(scale.reshape(-1).tolist()) & set(bias.reshape(-1).tolist())
                )
        if not report['selected_parameters']:
            raise ValueError('No coordinates selected; this is not a deletion candidate')
        # Both strategies get identical initialization RNG and minibatch order.
        torch.manual_seed(args.seed + 1000)
        reference = model_class()
        recovered = UnlearnNet(model.cpu(), selected,
                               reset_reference=reference if args.reset_strategy == 'initial' else None).to(device)
        generator = torch.Generator().manual_seed(args.seed + 2000)
        train_loader = DataLoader(Subset(train_data, retained_ids), batch_size=args.batch_size,
                                  shuffle=True, generator=generator)
        val_loader = DataLoader(Subset(train_data, validation_ids), batch_size=args.batch_size)
        report['history'] = recover(
            recovered, train_loader, val_loader, epochs=args.epochs,
            learning_rate=config['learning_rate'], momentum=config['momentum'],
            device=device, deadline=deadline, batchnorm_mode=args.batchnorm_mode,
        )
        final_model = model_class()
        final_model.load_state_dict(recovered.get_retrained_params())
        report['deletion_seconds'] = synchronized_time(device) - start
        report['cost_fraction_of_gold'] = report['deletion_seconds'] / gold_seconds
        if report['deletion_seconds'] >= report['budget_seconds']:
            raise TimeoutError('Deletion exceeded budget while exporting the recovered model')
        report_start = synchronized_time(device)
        report['utility'] = {}
        gold_test = read_pickle(folder / 'initial_eval_test_results.pkl')['shadow_out']['pred']
        gold_train = read_pickle(folder / 'initial_eval_train_results.pkl')['shadow_out']['pred']
        for name, dataset, truth, gold_prediction in [
            ('test', test_data, np.asarray(labels['test']), gold_test),
            ('retained', Subset(train_data, retained_ids), np.asarray(labels['train'])[retained_ids], gold_train[retained_ids]),
            ('forget', Subset(train_data, target_ids), np.asarray(labels['train'])[target_ids], gold_train[target_ids]),
        ]:
            result = evaluate_model(final_model, dataset, device, args.batch_size)
            np.savez(args.output / f'{name}_predictions.npz', **result)
            report['utility'][name] = {'accuracy': float(np.mean(result['pred'] == truth)),
                                       'disagreement_from_gold': float(np.mean(result['pred'] != gold_prediction)),
                                       'mean_true_class_margin': float(result['loss'].mean())}
        if bank is not None:
            dataset = Subset(ConcatDataset([train_data, test_data]), bank['candidate_complete_indices'].tolist())
            values = evaluate_lira(final_model, dataset, device, args.batch_size).astype(np.float64)
            np.save(args.output / 'lira_scores.npy', values)
            from sklearn.metrics import roc_auc_score, roc_curve

            def likelihood(mask):
                scores = bank['scores'].astype(np.float64)
                count = mask.sum(axis=0)
                mean = np.where(mask, scores, 0).sum(axis=0) / count
                variance = np.where(mask, (scores - mean) ** 2, 0).sum() / (count - 1).sum()
                std = max(float(np.sqrt(variance)), 1e-6)
                return -np.log(std) - 0.5 * ((values - mean) / std) ** 2

            llr = likelihood(bank['shadow_membership']) - likelihood(~bank['shadow_membership'])
            membership = bank['candidate_membership']
            fpr, tpr, _ = roc_curve(membership, llr)
            report['lira'] = {'auc': float(roc_auc_score(membership, llr)),
                              'tpr_at_fpr_0_01': float(tpr[fpr <= 0.01].max())}
        torch.save(final_model.state_dict(), args.output / 'recovered_model.pth')
        torch.save(selected, args.output / 'selected_coordinates.pt')
        report['reporting_seconds'] = synchronized_time(device) - report_start
        report['status'] = 'completed_pending_utility_and_privacy_review'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        report['elapsed_seconds_on_failure'] = synchronized_time(device) - start
        raise
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
