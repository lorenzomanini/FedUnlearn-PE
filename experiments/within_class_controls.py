"""C1/C2/diagonal/C4/C5 follow-ups on saved CIFAR-100 within-class runs.

No original or gold model is trained. Source artifacts are read-only and must
have been produced with --save-models. Only analysis results are saved: metrics,
per-example predictions and bit-packed masks of every recovered arm.

c1: fine-tune without reset (core 10% coordinates; all coordinates).
c2: layer-matched random resets at 20/30/40%, extending the random utility range.
diagonal: MC diagonal GGN ratio score at the source budgets.
c4: core, same-superclass swap and cross-superclass swap at other ranks, with a
    fixed number of subspace iterations; residuals are recorded, not gated.
c5: per-example comparison of every source checkpoint with the gold models.
"""

import argparse
import copy
import csv
import hashlib
import itertools
import json
import logging
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

from experiments import within_class as wc
from experiments.configs.cifar100_within_class import CONFIG as WITHIN_CLASS_CONFIG
from experiments.datasets import get_datasets
from experiments.models import create_resnet
from fisherunlearn import runtime
from fisherunlearn.information.diagonal import compute_client_information
from fisherunlearn.information.selection import (
    find_informative_params_matching, find_random_params_matching,
)
from fisherunlearn.unlearning import UnlearnNet, _flat_indices


STAGES = ('c5', 'c1', 'c2', 'diagonal', 'c4')


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def mask_overlap(first, second, parameters):
    """Compare coordinate sets, independent of serialization order."""
    shared = count_a = count_b = 0
    for name, parameter in parameters.items():
        empty = torch.empty((0, parameter.ndim), dtype=torch.long)
        a = _flat_indices(parameter, first.get(name, empty)).cpu()
        b = _flat_indices(parameter, second.get(name, empty)).cpu()
        shared += int(torch.isin(a, b).sum())
        count_a += a.numel()
        count_b += b.numel()
    union = count_a + count_b - shared
    return {'first_count': count_a, 'second_count': count_b, 'shared_count': shared,
            'shared_fraction': shared / count_a if count_a else None,
            'jaccard': shared / union if union else 1.,
            'identical': shared == count_a == count_b}


def _mask_key(mask, parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters.items():
        empty = torch.empty((0, parameter.ndim), dtype=torch.long)
        indices = _flat_indices(parameter, mask.get(name, empty)).cpu().sort().values
        digest.update(name.encode())
        digest.update(np.asarray([len(indices)], dtype=np.int64).tobytes())
        digest.update(indices.numpy().tobytes())
    return digest.hexdigest()


def _groups(test, protocol):
    fine, labels = np.asarray(test.fine_targets), np.asarray(test.targets)
    target = fine == protocol['config']['target_fine_class']
    siblings = (labels == protocol['target_superclass']) & ~target
    return {'target': target, 'siblings': siblings, 'retained': ~target,
            'other_superclasses': ~(target | siblings),
            'overall': np.ones(len(test), dtype=bool)}


def _csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_comparisons(predictions, gold_names, groups):
    """Descriptive pair distances; each seed remains one replicate."""
    pairs = [(a, b, 'gold_to_gold') for a, b in itertools.combinations(gold_names, 2)]
    pairs += [(a, b, 'candidate_to_gold') for a in predictions if a not in gold_names
              for b in gold_names]
    rows = []
    for a, b, kind in pairs:
        first, second = predictions[a], predictions[b]
        if not np.array_equal(first['labels'], second['labels']):
            raise ValueError('Prediction rows must have identical test labels/order')
        logits_a = torch.from_numpy(first['logits']).double()
        logits_b = torch.from_numpy(second['logits']).double()
        log_a, log_b = logits_a.log_softmax(1), logits_b.log_softmax(1)
        log_mid = torch.logaddexp(log_a, log_b) - np.log(2.)
        js = (.5 * ((log_a.exp() * (log_a - log_mid)).sum(1) +
                    (log_b.exp() * (log_b - log_mid)).sum(1))).numpy()
        centered_delta = ((logits_a - logits_a.mean(1, keepdim=True)) -
                          (logits_b - logits_b.mean(1, keepdim=True)))
        rms = centered_delta.square().mean(1).sqrt().numpy()
        disagreement = first['logits'].argmax(1) != second['logits'].argmax(1)
        for group, mask in groups.items():
            rows.append({'first': a, 'second': b, 'comparison': kind, 'group': group,
                         'count': int(mask.sum()), 'js_nats': float(js[mask].mean()),
                         'prediction_disagreement': float(disagreement[mask].mean()),
                         'centered_logit_rms': float(rms[mask].mean()),
                         'margin_mae': float(np.abs(first['margin'][mask] -
                                                     second['margin'][mask]).mean())})
    return rows


def prediction_summaries(predictions, gold_names, groups):
    """Mean/sign/quantiles expose why margin and accuracy closures differ."""
    rows = []
    for name, values in predictions.items():
        for group, mask in groups.items():
            margin, correct = values['margin'][mask], values['correct'][mask]
            gold_margin = np.mean([predictions[g]['margin'][mask].mean() for g in gold_names])
            gold_accuracy = np.mean([predictions[g]['correct'][mask].mean() for g in gold_names])
            original = predictions['original']
            rows.append({'model': name, 'group': group, 'count': int(mask.sum()),
                         'accuracy': float(correct.mean()), 'margin_mean': float(margin.mean()),
                         'loss_mean': float(values['loss'][mask].mean()),
                         'margin_q05': float(np.quantile(margin, .05)),
                         'margin_q50': float(np.quantile(margin, .5)),
                         'margin_q95': float(np.quantile(margin, .95)),
                         'margin_correct_mean': float(margin[correct].mean()) if correct.any() else None,
                         'margin_wrong_mean': float(margin[~correct].mean()) if (~correct).any() else None,
                         'margin_closure': wc.margin_gap_closure(
                             float(original['margin'][mask].mean()), float(margin.mean()), float(gold_margin)),
                         'accuracy_closure': wc.margin_gap_closure(
                             float(original['correct'][mask].mean()), float(correct.mean()), float(gold_accuracy))})
    return rows


def _validate_source(protocol, results, saved, train, test):
    config = protocol['config']
    if results['seed'] != protocol['seed']:
        raise ValueError('Source protocol and results seeds differ')
    if protocol['dataset_sizes'] != {'train': len(train), 'test': len(test)}:
        raise ValueError('Dataset sizes differ from the source run')
    for name, current in (('fine_classes', train.fine_classes), ('coarse_classes', train.classes),
                          ('fine_to_coarse', train.fine_to_coarse)):
        if list(protocol[name]) != list(current):
            raise ValueError(f'Dataset {name} differs from source')
    fine = np.asarray(train.fine_targets)
    target, retained, swap = (protocol[k] for k in ('target_indices', 'retained_indices', 'swap_indices'))
    expected_target = set(np.flatnonzero(fine == config['target_fine_class']).tolist())
    expected_swap = set(np.flatnonzero(fine == config['swap_fine_class']).tolist())
    if (set(target) != expected_target or set(swap) != expected_swap or
            set(retained) != set(range(len(train))) - expected_target or
            len(target) + len(retained) != len(train) or len(swap) != len(expected_swap)):
        raise ValueError('Source training split differs from the complete fine-class partition')
    allowed = {'full': set(range(len(train))), 'target': set(target), 'swap': set(swap)}
    for name, population in allowed.items():
        indices = saved['sample_indices'][name]
        if not indices or len(set(indices)) != len(indices) or not set(indices) <= population:
            raise ValueError(f'Invalid saved curvature sample: {name}')


def _diagonal(original, train, protocol, device):
    """Same MC diagonal ratio as CIFAR-10, on the complete training population."""
    config = protocol['config']
    wc._seed(protocol['seeds']['curvature'])
    original_state = wc._state(original)
    previous = runtime.DEVICE, runtime.INFO_BATCH_SIZE
    runtime.set_device(device)
    runtime.set_info_batch_size(config.get('spectral_batch_size', 32))
    try:
        scores = compute_client_information(
            1, original, nn.CrossEntropyLoss(),
            [Subset(train, protocol['retained_indices']), Subset(train, protocol['target_indices'])],
            method='diag_ggn_mc', stochastic_correction=False)
    finally:
        runtime.set_device(previous[0])
        runtime.set_info_batch_size(previous[1])
    parameters = {name: original_state[name] for name, _ in original.named_parameters()}
    if set(scores) != set(parameters) or any(scores[n].shape != p.shape for n, p in parameters.items()):
        raise ValueError('BackPACK conversion changed parameter names/shapes')
    return scores


def run(source, output, stages=STAGES, ranks=(40, 80), num_power_iters=None,
        cross_fine_class=None, device='cuda'):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose a new output directory')
    if source == output or source in output.parents:
        raise ValueError('Output must be outside the source experiment directory')
    if not stages or set(stages) - set(STAGES):
        raise ValueError(f'Stages must be a nonempty subset of {STAGES}')
    protocol = json.loads((source / 'protocol.json').read_text())
    baseline = json.loads((source / 'results.json').read_text())
    config = {**protocol['config'], 'download': False}
    if num_power_iters is None:
        num_power_iters = config['spectral_num_power_iters']
    if not ranks or len(set(ranks)) != len(ranks) or any(r < 1 for r in ranks) or num_power_iters < 0:
        raise ValueError('Ranks must be distinct positive integers; iterations must be nonnegative')
    wc._validate(config, device)
    if config['recovery_epochs'] != 1:
        raise ValueError('These controls require the source one-epoch recovery protocol')
    checkpoints = {'initial': source / 'initial.pt', 'original': source / 'original.pt'}
    gold_names = [f'gold_{i}' for i in range(len(baseline['golds']))]
    checkpoints.update({name: source / f'{name}.pt' for name in gold_names})
    original_masks = {}
    for arm in baseline['arms']:
        stem = f"{arm['arm']}_{arm['parameter_budget_percent']:g}"
        checkpoints[stem] = source / f'{stem}.pt'
        original_masks[stem] = source / f'{stem}_mask.pt'
    required = [source / n for n in ('protocol.json', 'results.json', 'scores.pt')]
    required += list(checkpoints.values()) + list(original_masks.values())
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'Missing source artifact: {path}. '
                                    'Rerun the source with SAVE_MODELS=1 sbatch queue.sh '
                                    'to run checkpoint-based controls.')
    if 'c1' in stages and 'core_10' not in original_masks:
        raise ValueError('C1 requires a saved 10% Core mask')
    saved = _load(source / 'scores.pt')
    train, test = get_datasets(config)
    _validate_source(protocol, baseline, saved, train, test)
    # Pilot runs predate the cross control; derive it from the fine labels.
    cross_indices = protocol.get('cross_indices')
    if cross_indices is None:
        cross = WITHIN_CLASS_CONFIG['cross_fine_class'] if cross_fine_class is None else cross_fine_class
        fine = np.asarray(train.fine_targets)
        if train.fine_to_coarse[cross] == protocol['target_superclass']:
            raise ValueError('cross control must come from a different superclass than the target')
        cross_indices = np.flatnonzero(fine == cross).tolist()
        if not cross_indices or not set(cross_indices) <= set(protocol['retained_indices']):
            raise ValueError('cross fine-class records must all be retained')
    else:
        cross = protocol['config']['cross_fine_class']
    original = create_resnet(config).cpu().eval()
    original.load_state_dict(_load(checkpoints['original']))
    initial = _load(checkpoints['initial'])
    original_state = wc._state(original)
    parameters = dict(original.named_parameters())
    masks = {name: _load(path) for name, path in original_masks.items()}
    for name, scores in saved['scores'].items():
        if set(scores) != set(parameters) or any(scores[n].shape != p.shape for n, p in parameters.items()):
            raise ValueError(f'Saved {name} scores differ from model parameter names/shapes')
    for mask in masks.values():
        if set(mask) - set(parameters):
            raise ValueError('Saved mask contains unknown parameters')
        mask_overlap(mask, mask, parameters)  # validates bounds and uniqueness
    output.mkdir(parents=True)
    source_snapshot = output / 'source'
    root = Path(__file__).resolve().parents[1]
    for package in ('experiments', 'fisherunlearn'):
        for path in (root / package).rglob('*.py'):
            destination = source_snapshot / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    report = {'source': str(source), 'seed': protocol['seed'], 'stages': list(stages),
              'save_models': False,
              'ranks': list(ranks), 'num_power_iters': num_power_iters,
              'convergence': 'not required; residuals recorded per rank', 'config': config,
              'cross_fine_class': int(cross), 'cross_fine_name': train.fine_classes[cross],
              'c1_budget': 10., 'random_budgets': [20., 30., 40.],
              'diagonal': 'diag_ggn_mc; complete training population; global scalar budgets',
              'recovery_seed': protocol['seeds']['recovery'],
              'source_sha256': {str(p.relative_to(source)): _hash(p) for p in required},
              'runtime': {'torch': torch.__version__, 'numpy': np.__version__,
                          'device': str(device), 'cuda': torch.version.cuda,
                          'gpu': torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else None},
              'status': 'running', 'arms': [], 'mask_comparisons': [], 'scoring': {},
              'score_summaries': {}}
    wc._write_json(output / 'protocol.json', protocol)
    wc._write_json(output / 'results.json', report)
    predictions = {}
    groups = _groups(test, protocol)
    (output / 'predictions').mkdir()
    retained = Subset(wc.training_view(train, config), protocol['retained_indices'])
    shapes = {name: tuple(p.shape) for name, p in parameters.items()}
    control_masks = {}

    def reuse_prediction(name):
        """Newer source runs already saved test logits; don't duplicate them."""
        path = source / 'predictions' / f'{name}.npz'
        if not path.is_file():
            return False
        with np.load(path) as stored:
            values = {key: stored[key] for key in stored.files}
        if not np.array_equal(values['labels'], np.asarray(test.targets)):
            raise ValueError(f'Saved predictions for {name} do not follow official test order')
        predictions[name] = values
        return True

    def predict(name, model):
        logging.info('Saving logits for %s', name)
        values = wc._predict(model, test, device, config['eval_batch_size'], return_logits=True)
        if not np.array_equal(values['labels'], np.asarray(test.targets)):
            raise ValueError('Prediction labels do not follow official test order')
        if not np.isfinite(values['logits']).all():
            raise ValueError(f'Nonfinite logits for {name}')
        values.update(test_index=np.arange(len(test)), fine_labels=np.asarray(test.fine_targets),
                      member_index=np.asarray(protocol['target_indices']),
                      member_loss=wc._predict(model, Subset(train, protocol['target_indices']),
                                              device, config['eval_batch_size'])['loss'])
        np.savez_compressed(output / 'predictions' / f'{name}.npz', **values)
        predictions[name] = values

    def write_comparisons():
        _csv(output / 'prediction_comparisons.csv', prediction_comparisons(predictions, gold_names, groups))
        _csv(output / 'prediction_summary.csv', prediction_summaries(predictions, gold_names, groups))

    # Golds and original anchor every stage's prediction-level diagnostics.
    evaluation = gold_names + ['original']
    if 'c5' in stages:
        evaluation += [name for name in checkpoints if name not in evaluation]
    for name in evaluation:
        if reuse_prediction(name):
            continue
        model = copy.deepcopy(original)
        model.load_state_dict(_load(checkpoints[name]))
        predict(name, model)
        del model
    write_comparisons()

    def recover(name, mask=None, reset=True, budget=None, reuse=None):
        logging.info('Recovery %s (reuse=%s)', name, reuse)
        started = time.perf_counter()
        if reuse is not None:
            model = copy.deepcopy(original)
            model.load_state_dict(_load(reuse))
        elif mask is None:
            model = copy.deepcopy(original)
            wc._train(model, retained, 1, config, device, protocol['seeds']['recovery'])
        else:
            net = UnlearnNet(original, mask, reset_reference=initial if reset else original_state)
            wc._train(net, retained, 1, config, device, protocol['seeds']['recovery'])
            model = copy.deepcopy(original)
            model.load_state_dict(net.get_retrained_params())
            del net
        audit = None
        if mask is not None:
            control_masks[name] = mask
            if reuse is None:  # reused checkpoints were audited with their source run
                audit = wc._audit(wc._state(model), original_state, shapes, wc.mask_bits(mask, shapes))
        metrics = wc._metrics(model, train, test, protocol['target_indices'], config, device)
        predict(name, model)
        write_comparisons()
        report['arms'].append({'arm': name, 'parameter_budget_percent': budget,
                               'selected_count': sum(map(len, mask.values())) if mask is not None else
                                                 sum(p.numel() for p in parameters.values()),
                               'reset': reset, 'reused_checkpoint': str(reuse) if reuse else None,
                               'audit': audit, 'metrics': metrics,
                               'elapsed_seconds': time.perf_counter() - started})
        wc._write_json(output / 'results.json', report)

    def flush_masks():
        if control_masks:
            wc.save_masks(output / 'masks.npz', control_masks, shapes)

    if 'c1' in stages:
        recover('no_reset_core_10', masks['core_10'], reset=False, budget=10.)
        recover('no_reset_all', reset=False, budget=100.)
    if 'c2' in stages:
        for index, budget in enumerate((20., 30., 40.)):
            core = wc.select_global_parameters(saved['scores']['core'], budget)
            wc._seed(protocol['seeds']['random_selection'] + len(config['parameter_budgets']) + index)
            mask = find_random_params_matching(saved['scores']['core'], core)
            recover(f'random_{budget:g}', mask, budget=budget)
        flush_masks()
    if 'diagonal' in stages:
        started = time.perf_counter()
        scores = _diagonal(original, train, protocol, device)
        report['scoring']['diagonal'] = {'elapsed_seconds': time.perf_counter() - started}
        for budget in config['parameter_budgets']:
            recover(f'diagonal_{budget:g}', wc.select_global_parameters(scores, budget), budget=budget)
        flush_masks()
    if 'c4' in stages:
        # Reuse matching source checkpoints; new models are evaluated and discarded.
        available = {_mask_key(masks[name], parameters): checkpoints[name] for name in masks
                     if name.startswith(('core_', 'swap_', 'cross_'))}
        source_rank = config['spectral_rank']
        previous_rank = {}
        for rank in ranks:
            # A fixed iteration count at every rank: the derivation needs only
            # U^T H U > 0, so an unconverged subspace is valid; residuals are kept.
            logging.info('C4: rank %s, %s subspace iterations, no convergence gate',
                         rank, num_power_iters)
            scoring_config = {**config, 'spectral_rank': rank, 'spectral_num_power_iters': num_power_iters}
            scores, diagnostics = wc._score(
                original, train, protocol['target_indices'], protocol['swap_indices'],
                scoring_config, protocol['seeds']['curvature'], device,
                sample_indices=saved['sample_indices'], cross_indices=cross_indices)
            report['scoring'][str(rank)] = diagnostics
            report['score_summaries'][str(rank)] = wc.summarize_scores(scores)
            wc._write_json(output / 'results.json', report)
            for budget in config['parameter_budgets']:
                core = wc.select_global_parameters(scores['core'], budget)
                ranked = {'core': core,
                          'swap': find_informative_params_matching(scores['swap'], core),
                          'cross': find_informative_params_matching(scores['cross'], core)}
                for first, second in (('core', 'swap'), ('core', 'cross'), ('swap', 'cross')):
                    report['mask_comparisons'].append({
                        'rank': rank, 'budget': budget, 'comparison': f'{first}_vs_{second}',
                        **mask_overlap(ranked[first], ranked[second], parameters)})
                for arm, mask in ranked.items():
                    old = masks.get(f'{arm}_{budget:g}')
                    if old is not None:
                        report['mask_comparisons'].append({
                            'rank': rank, 'budget': budget,
                            'comparison': f'{arm}_vs_source_rank{source_rank}',
                            **mask_overlap(mask, old, parameters)})
                    key = arm, budget
                    if key in previous_rank:
                        old_rank, previous_mask = previous_rank[key]
                        report['mask_comparisons'].append({
                            'rank': rank, 'previous_rank': old_rank, 'budget': budget,
                            'comparison': f'{arm}_vs_previous_rank',
                            **mask_overlap(mask, previous_mask, parameters)})
                    previous_rank[key] = rank, mask
                    reuse = available.get(_mask_key(mask, parameters))
                    recover(f'rank_{rank}_{arm}_{budget:g}', mask, budget=budget, reuse=reuse)
            flush_masks()
    flush_masks()
    report['status'] = 'complete'
    wc._write_json(output / 'results.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='saved seed_*/experiment directory')
    parser.add_argument('--output', required=True, type=Path, help='new directory, outside source experiment')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--ranks', nargs='+', type=int, default=[40, 80])
    parser.add_argument('--num-power-iters', type=int, default=None,
                        help='fixed subspace iterations at every c4 rank (default: source protocol)')
    parser.add_argument('--cross-fine-class', type=int, default=None,
                        help='cross-superclass control for sources that predate it (default: config)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    run(args.source, args.output, args.stages, args.ranks, args.num_power_iters,
        args.cross_fine_class, args.device)


if __name__ == '__main__':
    main()
