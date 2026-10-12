"""Fixed-checkpoint CIFAR mask and recovery controls, using an existing LiRA bank.

Research ablation: costs are reported, without the cheap-deletion budget cutoff.
Run ``python -m experiments.cifar_diagnosis --help`` or submit queue_diagnosis.sh.
Use --reuse-diagnosis JOB_DIR for the four full-weight recovery controls without
training a new original or recomputing curvature (queue_recovery_controls.sh).
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score, roc_curve

from analysis.results import fit_lira_distributions
from experiments.cifar_recovery_ablation import read_pickle, recover, synchronized_time
from experiments.datasets import get_datasets
from experiments.evaluation import evaluate_lira, evaluate_model
from experiments.models import get_model_class
from experiments.runner import _curvature_loader
from fisherunlearn.information.selection import (
    find_informative_params, find_informative_params_matching, find_random_params_matching,
)
from fisherunlearn.information.spectral_wip import (
    core_score_from_subspace, estimate_core_score, make_hessian_matvec, unflatten_like,
)
from fisherunlearn.unlearning import UnlearnNet, reset_parameters, _flat_indices


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def compare_masks(scores, first, second):
    """Micro Jaccard and retention, plus overlap for each parameter tensor."""
    rows = {}
    for name, tensor in scores.items():
        a = _flat_indices(tensor, first[name])
        b = _flat_indices(tensor, second[name])
        intersection = int(torch.isin(a, b).sum())
        rows[name] = dict(first_count=len(a), second_count=len(b), intersection=intersection,
                          union=len(a) + len(b) - intersection)
    totals = {key: sum(row[key] for row in rows.values()) for key in next(iter(rows.values()))}
    return {**totals, 'jaccard': totals['intersection'] / max(totals['union'], 1),
            'first_mask_retained_fraction': totals['intersection'] / max(totals['first_count'], 1),
            'per_tensor': rows}


def privacy_metrics(values, bank, distributions):
    values = np.asarray(values, dtype=np.float64)
    membership = bank['candidate_membership']
    if values.shape != membership.shape or not np.isfinite(values).all():
        raise ValueError('LiRA statistics must be finite and match candidate order')
    inside, outside = distributions
    llr = -np.log(inside['std']) - .5 * ((values - inside['avg']) / inside['std']) ** 2
    llr += np.log(outside['std']) + .5 * ((values - outside['avg']) / outside['std']) ** 2
    fpr, tpr, _ = roc_curve(membership, llr)
    return dict(lira_auc=float(roc_auc_score(membership, llr)),
                lira_tpr_at_fpr_0_001=float(tpr[fpr <= .001].max()),
                lira_tpr_at_fpr_0_01=float(tpr[fpr <= .01].max()),
                raw_confidence_auc=float(roc_auc_score(membership, values)),
                raw_confidence_gap=float(values[membership].mean() - values[~membership].mean())), llr


def run(args):
    parent = (json.loads((args.reuse_diagnosis / 'controls/report.json').read_text())
              if args.reuse_diagnosis else None)
    config = read_pickle(args.suite / 'init_params.pkl')
    if config['dataset_name'] != 'cifar10' or config['model_name'] != 'resnet18':
        raise ValueError('Expected CIFAR10/ResNet18 suite')
    folder = args.suite / f'test_{args.repetition}'
    split = read_pickle(folder / 'score_diagnostics.pkl')
    labels = {key: np.asarray(value) for key, value in read_pickle(args.suite / 'labels.pkl').items()}
    train_data, test_data = get_datasets(config)
    for dataset, key in ((train_data, 'train'), (test_data, 'test')):
        if not np.array_equal(np.asarray(dataset.targets), labels[key]):
            raise ValueError(f'{key} label order differs from saved suite')
    train_ids, target_ids = list(split['training_indices']), list(split['target_indices'])
    if (len(train_ids) != len(set(train_ids)) or len(target_ids) != len(set(target_ids))
            or not target_ids or not set(target_ids).issubset(train_ids)
            or any(i < 0 or i >= len(train_data) for i in train_ids)):
        raise ValueError('Invalid saved split')
    target_set = set(target_ids)
    retained_ids = [i for i in train_ids if i not in target_set]
    with np.load(args.suite / 'lira_shadow_bank.npz') as archive:
        bank = dict(archive)
    candidates = bank['candidate_complete_indices']
    membership = bank['candidate_membership']
    if (membership.dtype != bool or candidates.shape != membership.shape
            or candidates.ndim != 1 or len(np.unique(candidates)) != len(candidates)
            or not np.any(~membership) or set(candidates[membership]) != set(target_ids)
            or any(i < 0 or i >= len(train_data) + len(test_data) for i in candidates)
            or set(candidates[~membership]) & set(train_ids)
            or not np.isfinite(bank['scores']).all()):
        raise ValueError('LiRA candidates/membership do not match the saved split')
    validation_ids = sorted(set(range(len(train_data))) - set(train_ids) - set(candidates))
    if len(retained_ids) < len(target_ids) or not validation_ids:
        raise ValueError('Need an equal-sized retained pseudo-target and unaudited validation records')
    if parent is not None:
        parent_split = torch.load(args.reuse_diagnosis / 'controls/split.pt', map_location='cpu', weights_only=False)
        for key, indices in [('training_indices', train_ids), ('target_indices', target_ids),
                             ('retained_indices', retained_ids), ('validation_indices', validation_ids)]:
            if list(parent_split[key]) != indices:
                raise ValueError(f'Saved diagnosis {key} differs from the supplied suite')
        pseudo_ids = list(parent_split['pseudo_target_indices'])
        if len(set(pseudo_ids)) != len(target_ids) or not set(pseudo_ids).issubset(retained_ids):
            raise ValueError('Saved pseudo-target must be unique, retained and equal-sized')
        samples = parent_split['curvature_sample_indices']
    else:
        pseudo_order = torch.randperm(len(retained_ids), generator=torch.Generator().manual_seed(args.seed + 2))
        pseudo_ids = [retained_ids[i] for i in pseudo_order[:len(target_ids)].tolist()]
        datasets = {'full': Subset(train_data, train_ids), 'target': Subset(train_data, target_ids),
                    'pseudo': Subset(train_data, pseudo_ids)}
        loaders = {key: _curvature_loader(data, args.max_samples, args.batch_size,
                                        args.seed + (key != 'full')) for key, data in datasets.items()}
        samples = {key: [datasets[key].indices[i] for i in loader.dataset.indices]
                   for key, loader in loaders.items()}

    model_class = get_model_class(config)
    model = model_class().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    initial = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=True)
    if set(initial) != set(model.state_dict()) or any(initial[k].shape != v.shape for k, v in model.state_dict().items()):
        raise ValueError('Initialization must have the same state structure as the original checkpoint')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    check_ids = train_ids[:128]
    check = evaluate_model(model, Subset(train_data, check_ids), device, args.batch_size)
    original_predictions = read_pickle(folder / 'initial_eval_train_results.pkl')['trained']
    if not np.array_equal(check['pred'], original_predictions['pred'][check_ids]) or not np.allclose(
            check['loss'], original_predictions['loss'][check_ids], rtol=1e-3, atol=1e-3):
        raise ValueError('Checkpoint does not match saved original predictions')
    gold_train = read_pickle(folder / 'initial_eval_train_results.pkl')['shadow_out']
    gold_test = read_pickle(folder / 'initial_eval_test_results.pkl')['shadow_out']
    distributions = fit_lira_distributions(bank['scores'], bank['shadow_membership'],
                                           global_var=bool(config.get('lira_global_variance', True)))
    audit_dataset = Subset(ConcatDataset([train_data, test_data]), candidates.tolist())
    provenance = {key: file_hash(path) for key, path in
                  [('checkpoint_sha256', args.checkpoint), ('initialization_sha256', args.initial_checkpoint),
                   ('shadow_bank_sha256', args.suite / 'lira_shadow_bank.npz')]}
    reused_masks = {}
    mask_hashes = {}
    if parent is not None:
        if provenance != parent['provenance']:
            raise ValueError('Checkpoint, initialization or LiRA bank differs from the source diagnosis')
        if parent['retained_batch_seed'] != args.seed + 2000:
            raise ValueError('Retained batch seed differs from the source diagnosis')
        parameters = dict(model.named_parameters())
        for name in ('actual_selected', 'pseudo_selected', 'random_selected'):
            path = args.reuse_diagnosis / 'controls' / f'{name}_mask.pt'
            mask = torch.load(path, map_location='cpu', weights_only=True)
            if set(mask) != set(parameters):
                raise ValueError(f'{name} mask parameter names differ from the checkpoint')
            counts = {key: len(indices) for key, indices in mask.items()}
            if counts != parent['conditions'][name]['selected_per_tensor'] or not sum(counts.values()):
                raise ValueError(f'{name} mask counts differ from the source diagnosis')
            for key, indices in mask.items():
                _flat_indices(parameters[key], indices)  # Bounds and uniqueness.
            reused_masks[name] = mask
            mask_hashes[name] = file_hash(path)
        layer_counts = []
        for mask in reused_masks.values():
            counts = {}
            for key, indices in mask.items():
                layer = key.rpartition('.')[0]
                counts[layer] = counts.get(layer, 0) + len(indices)
            layer_counts.append(counts)
        if not layer_counts[0] == layer_counts[1] == layer_counts[2]:
            raise ValueError('Source actual/pseudo/random masks must match layer counts')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  mode='recovery_controls' if parent is not None else 'diagnosis', provenance=provenance,
                  split={**split, 'retained_indices': retained_ids, 'validation_indices': validation_ids,
                         'pseudo_target_indices': pseudo_ids, 'curvature_sample_indices': samples},
                  scores={}, mask_comparisons={}, rows=[], conditions={}, total_auditing_seconds=0.,
                  retained_batch_seed=args.seed + 2000,
                  optimizer=dict(name='SGD', learning_rate=config['learning_rate'], momentum=config['momentum']),
                  note='One fixed original checkpoint. Previous gold predictions are a descriptive reference. '
                       'All privacy audits use the actual target, including pseudo-target controls. '
                       'Research cost is not a bounded deletion-cost claim.')
    if parent is not None:
        report['source_diagnosis'] = dict(directory=str(args.reuse_diagnosis),
                                         report_sha256=file_hash(args.reuse_diagnosis / 'controls/report.json'),
                                         split_sha256=file_hash(args.reuse_diagnosis / 'controls/split.pt'),
                                         mask_sha256=mask_hashes,
                                         specificity_status=parent.get('specificity_status'),
                                         previous_actual_full_rows=[row for row in parent['rows'] if row['condition'] == 'actual_full'])
    # Keep split arrays in a separate file; they may be NumPy values in supplied suites.
    torch.save(report.pop('split'), args.output / 'split.pt')
    started = synchronized_time(device)

    def save_report():
        report['elapsed_seconds'] = synchronized_time(device) - started
        temporary = args.output / 'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output / 'report.json')
        if report['rows']:
            with (args.output / 'metrics.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(report['rows'][0]))
                writer.writeheader()
                writer.writerows(report['rows'])

    def audit(state, condition, epoch, validation=None, recovery_seconds=0.):
        audit_start = synchronized_time(device)
        output = args.output / condition / f'epoch_{epoch}'
        output.mkdir(parents=True)
        candidate = model_class().eval()
        candidate.load_state_dict(state)
        torch.save(state, output / 'model.pth')
        train_result = evaluate_model(candidate, train_data, device, args.batch_size)
        test_result = evaluate_model(candidate, test_data, device, args.batch_size)
        np.savez(output / 'train_predictions.npz', **train_result)
        np.savez(output / 'test_predictions.npz', **test_result)
        row = dict(condition=condition, epoch=epoch, validation_accuracy=validation,
                   recovery_seconds=recovery_seconds)
        for key, result, truth, gold, indices in [
            ('test', test_result, labels['test'], gold_test, slice(None)),
            ('retained', train_result, labels['train'], gold_train, retained_ids),
            ('forget', train_result, labels['train'], gold_train, target_ids),
        ]:
            row[key + '_accuracy'] = float(np.mean(result['pred'][indices] == truth[indices]))
            row[key + '_disagreement_from_previous_gold'] = float(np.mean(result['pred'][indices] != gold['pred'][indices]))
        values = evaluate_lira(candidate, audit_dataset, device, args.batch_size)
        metrics, llr = privacy_metrics(values, bank, distributions)
        np.savez(output / 'privacy_scores.npz', confidence=values, llr=llr)
        row.update(metrics)
        row['audit_seconds'] = synchronized_time(device) - audit_start
        report['total_auditing_seconds'] += row['audit_seconds']
        report['rows'].append(row)
        save_report()
        print(f"{condition} epoch {epoch}: test={row['test_accuracy']:.4f}, LiRA={row['lira_auc']:.4f}", flush=True)

    try:
        save_report()
        audit({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 'original', 0)
        gold_row = dict(report['rows'][0], condition='previous_gold', validation_accuracy=None,
                        recovery_seconds=0., audit_seconds=0.,
                        **{key: None for key in report['rows'][0] if key.startswith(('lira_', 'raw_'))})
        for key, result, truth, indices in [('test', gold_test, labels['test'], slice(None)),
                                          ('retained', gold_train, labels['train'], retained_ids),
                                          ('forget', gold_train, labels['train'], target_ids)]:
            gold_row[key + '_accuracy'] = float(np.mean(result['pred'][indices] == truth[indices]))
            gold_row[key + '_disagreement_from_previous_gold'] = 0.
        previous_gold = folder / 'previous_gold_lira_scores.npy'
        if previous_gold.exists():
            gold_row.update(privacy_metrics(np.load(previous_gold), bank, distributions)[0])
        report['rows'].append(gold_row)
        if parent is not None:
            empty = {name: torch.empty((0, parameter.ndim), dtype=torch.long)
                     for name, parameter in model.named_parameters()}
            conditions = [('no_reset_full', empty, True, 'train'),
                          ('actual_full', reused_masks['actual_selected'], True, 'train'),
                          ('pseudo_full', reused_masks['pseudo_selected'], True, 'train'),
                          ('random_full', reused_masks['random_selected'], True, 'train')]
        else:
            score_directory = args.output / 'scores'
            score_directory.mkdir()
            thresholds = dict(eigenvalue_threshold=config.get('spectral_eigenvalue_min', 1e-6),
                              relative_eigenvalue_threshold=config.get('spectral_eigenvalue_rtol', 1e-5))
            masks = {}
            reference = None
            for name, rank, iterations in [('baseline', args.rank, args.baseline_power_iters),
                                           ('reference', args.rank, args.power_iters),
                                           ('higher_rank', args.higher_rank, args.power_iters)]:
                print(f'Scoring {name}: rank={rank}, maximum updates={iterations}', flush=True)
                torch.manual_seed(args.seed)
                score = estimate_core_score(model.to(device), loaders['full'], loaders['target'],
                                            nn.CrossEntropyLoss(), rank, device, num_power_iters=iterations,
                                            power_tolerance=args.power_tolerance, return_evecs=name == 'reference', **thresholds)
                masks[name] = find_informative_params(score['diag_by_name'], 'information', args.score_mass)
                torch.save({key: value for key, value in score.items() if key != 'evecs'}, score_directory / f'{name}.pt')
                torch.save(masks[name], score_directory / f'{name}_mask.pt')
                report['scores'][name] = score['diagnostics']
                if name == 'reference':
                    reference = score
                save_report()
            report['mask_comparisons']['baseline_vs_reference'] = compare_masks(reference['diag_by_name'], masks['baseline'], masks['reference'])
            report['mask_comparisons']['reference_vs_higher_rank'] = compare_masks(reference['diag_by_name'], masks['reference'], masks['higher_rank'])
            report['specificity_status'] = ('exploratory_until_mask_stability_review' if
                                           not reference['diagnostics']['subspace']['converged'] else
                                           'converged_reference_still_requires_mask_stability_review')

            # Compare target identity in precisely the same H basis, without rerunning H.
            pseudo_start = synchronized_time(device)
            operator, shapes, numels, _ = make_hessian_matvec(model, loaders['pseudo'], nn.CrossEntropyLoss(), device)
            basis = reference.pop('evecs').to(device)
            products = operator(basis)
            projection = basis.T @ products
            del products
            pseudo = core_score_from_subspace(basis, reference['evals_H'].to(device), projection, **thresholds)
            pseudo_scores = dict(zip(reference['diag_by_name'], unflatten_like(pseudo['diag_flat'].cpu(), shapes, numels)))
            torch.save({'diag_by_name': pseudo_scores, 'projected_target': pseudo['projected_target'].cpu(),
                        'whitened_target': pseudo['whitened_target'].cpu(), 'diagnostics': pseudo['diagnostics']},
                       score_directory / 'pseudo.pt')
            report['scores']['pseudo'] = {**pseudo['diagnostics'], 'same_basis_as': 'reference',
                                          'elapsed_seconds': synchronized_time(device) - pseudo_start}
            actual = masks['reference']
            pseudo_mask = find_informative_params_matching(pseudo_scores, actual)
            torch.manual_seed(args.seed + 3)
            random_mask = find_random_params_matching(reference['diag_by_name'], actual)
            report['mask_comparisons']['actual_vs_pseudo'] = compare_masks(reference['diag_by_name'], actual, pseudo_mask)
            report['mask_comparisons']['actual_vs_random'] = compare_masks(reference['diag_by_name'], actual, random_mask)
            del basis, pseudo, projection, reference, score, operator
            model.cpu()
            bn_names = {f'{name}.{suffix}' for name, module in model.named_modules()
                        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.affine
                        for suffix in ('weight', 'bias')}
            no_bn = {name: indices if name not in bn_names else indices[:0] for name, indices in actual.items()}
            # This control changes mask size; it is not count-matched evidence of causality.
            conditions = [('actual_selected', actual, False, 'train'),
                          ('pseudo_selected', pseudo_mask, False, 'train'),
                          ('random_selected', random_mask, False, 'train'),
                          ('actual_full', actual, True, 'train'),
                          ('actual_frozen_bn', actual, False, 'frozen'),
                          ('actual_keep_bn_affine', no_bn, False, 'train')]
        report['total_scoring_seconds'] = sum(item['elapsed_seconds'] for item in report['scores'].values())
        for name, mask, full, bn_mode in conditions:
            reset_count = sum(len(indices) for indices in mask.values())
            print(f'Recovery control {name}: full_weights={full}, reset_scalars={reset_count}', flush=True)
            if name == 'no_reset_full':
                if not full or reset_count != 0:
                    raise ValueError('no_reset_full requires full-weight recovery and an empty reset mask')
                # Continue training from the original checkpoint, without a reset.
                reset = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                if reset_count == 0:
                    raise ValueError(f'{name} selects no parameters')
                reset = reset_parameters(model, mask, initial)
            torch.save(mask, args.output / f'{name}_mask.pt')
            report['conditions'][name] = dict(selected_per_tensor={key: len(value) for key, value in mask.items()},
                                             trainable='all' if full else 'selected', batchnorm_mode=bn_mode)
            audit(reset, name, 0)
            torch.manual_seed(args.seed + 2000)
            recovered = model_class() if full else UnlearnNet(model, mask, reset_reference=initial)
            if full:
                recovered.load_state_dict(reset)
            recovered.to(device)
            train_loader = DataLoader(Subset(train_data, retained_ids), batch_size=args.batch_size,
                                      shuffle=True, generator=torch.Generator().manual_seed(args.seed + 2000))
            validation_loader = DataLoader(Subset(train_data, validation_ids), batch_size=args.batch_size)
            recovery_start, auditing_seconds = synchronized_time(device), 0.

            def on_epoch(epoch, trained, history):
                nonlocal auditing_seconds
                audit_start = synchronized_time(device)
                state = (trained.get_retrained_params() if isinstance(trained, UnlearnNet)
                         else {key: value.detach().cpu().clone() for key, value in trained.state_dict().items()})
                audit(state, name, epoch, history['validation_accuracy'],
                      audit_start - recovery_start - auditing_seconds)
                auditing_seconds += synchronized_time(device) - audit_start

            history = recover(recovered, train_loader, validation_loader, epochs=args.epochs,
                              learning_rate=config['learning_rate'], momentum=config['momentum'],
                              device=device, deadline=float('inf'), batchnorm_mode=bn_mode, on_epoch=on_epoch)
            report['conditions'][name]['history'] = history
            report['conditions'][name]['best_epoch_by_retained_validation'] = max(history, key=lambda row: row['validation_accuracy'])['epoch']
            report['conditions'][name]['recovery_seconds_excluding_audit'] = synchronized_time(device) - recovery_start - auditing_seconds
            del recovered
            save_report()
        report['status'] = 'completed_diagnostic_controls_pending_analysis'
    except Exception as error:
        report['status'], report['error'] = 'failed', str(error)
        raise
    finally:
        save_report()
    print(f'Results saved in {args.output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reuse-diagnosis', type=Path,
                        help='Completed diagnosis job directory containing bootstrap/ and controls/. Reuse exact masks for four full-weight controls.')
    parser.add_argument('--initial-checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetition', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--higher-rank', type=int, default=40)
    parser.add_argument('--baseline-power-iters', type=int, default=5)
    parser.add_argument('--power-iters', type=int, default=25)
    parser.add_argument('--power-tolerance', type=float, default=.001)
    parser.add_argument('--max-samples', type=int, default=2048)
    parser.add_argument('--score-mass', type=float, default=100 * 5 / 9)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    if args.reuse_diagnosis:
        if args.suite or args.checkpoint or args.initial_checkpoint:
            parser.error('--reuse-diagnosis derives suite and checkpoints; omit their separate arguments')
        source_report = args.reuse_diagnosis / 'controls/report.json'
        if not source_report.is_file():
            parser.error('--reuse-diagnosis must contain controls/report.json')
        parent = json.loads(source_report.read_text())
        if parent.get('status') != 'completed_diagnostic_controls_pending_analysis' or parent.get('mode') == 'recovery_controls':
            parser.error('--reuse-diagnosis requires a completed original diagnosis pilot')
        saved = parent['arguments']
        for key in ('seed', 'repetition', 'batch_size'):
            value = getattr(args, key)
            if value is not None and value != saved[key]:
                parser.error(f'--{key.replace("_", "-")} must match the source diagnosis')
            setattr(args, key, saved[key])
        args.epochs = args.epochs if args.epochs is not None else saved['epochs']
        args.suite = args.reuse_diagnosis / 'bootstrap/suite'
        args.checkpoint = args.reuse_diagnosis / 'bootstrap/original_model.pth'
        args.initial_checkpoint = args.reuse_diagnosis / 'bootstrap/initial_model.pth'
        # Scoring settings describe the saved masks only; this mode never scores.
        for key in ('rank', 'higher_rank', 'baseline_power_iters', 'power_iters', 'power_tolerance', 'max_samples', 'score_mass'):
            setattr(args, key, saved[key])
    else:
        if args.suite is None or args.checkpoint is None:
            parser.error('--suite and --checkpoint are required without --reuse-diagnosis')
        for key, default in [('repetition', 0), ('seed', 0), ('batch_size', 128), ('epochs', 5)]:
            if getattr(args, key) is None:
                setattr(args, key, default)
    args.initial_checkpoint = args.initial_checkpoint or args.checkpoint.with_name('initial_model.pth')
    if args.output.exists():
        parser.error('--output must be a new directory')
    if not args.checkpoint.is_file() or not args.initial_checkpoint.is_file():
        parser.error('Original and matching initialization checkpoints must exist')
    if (min(args.rank, args.max_samples, args.epochs, args.batch_size) < 1 or args.repetition < 0
            or args.higher_rank <= args.rank or args.baseline_power_iters < 0
            or args.power_iters <= args.baseline_power_iters):
        parser.error('Positive sizes, nonnegative repetition/iterations, larger comparison rank and iteration budget required')
    if (not math.isfinite(args.score_mass) or not 0 < args.score_mass <= 100
            or not math.isfinite(args.power_tolerance) or args.power_tolerance <= 0):
        parser.error('Score mass must be in (0,100]; tolerance must be finite and positive')
    runner = Path(__file__).resolve()
    print(f'Diagnosis runner: {runner}; SHA256: {file_hash(runner)}', flush=True)
    run(args)


if __name__ == '__main__':
    main()
