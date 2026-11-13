"""Forget one CIFAR-100 fine class while predicting its retained superclass."""

import copy
import itertools
import json
import logging
import math
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.metrics import roc_auc_score

from experiments.datasets import get_clients_subsets, get_datasets
from experiments.models import create_resnet
from fisherunlearn.information.selection import (
    find_informative_params_matching, find_random_params_matching,
)
from fisherunlearn.information.spectral_wip import (
    core_score_from_subspace, estimate_core_score, make_ggn_matvec, score_agreement,
)
from fisherunlearn.unlearning import UnlearnNet, _flat_indices


def _seed(value):
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _json(value):
    if torch.is_tensor(value):
        return _json(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(_json(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def training_view(train, config):
    """Records used for SGD; scoring, member losses and evaluation keep ``train``."""
    if not config.get('augment', False):
        return train
    from torchvision import transforms

    if getattr(train, 'transform', None) is None:
        raise ValueError('augmentation requires an image dataset with a transform')
    view = copy.copy(train)
    # Cropping 32 px with 4 px padding before the 64 px resize equals 64/8 after it.
    view.transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(), train.transform])
    return view


def _train(model, dataset, epochs, config, device, seed):
    """Keep every record; explicit generators pair minibatches across reset arms."""
    _seed(seed)
    model.to(device)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],
                                momentum=config.get('momentum', 0.9),
                                weight_decay=config.get('weight_decay', 0.0))
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.numel()
        logging.info('Training seed %s epoch %s/%s, loss %.6f',
                     seed, epoch + 1, epochs, total_loss / len(dataset))
    return model.eval()


def _predict(model, dataset, device, batch_size, return_logits=False):
    model.to(device).eval()
    outputs, labels = [], []
    with torch.no_grad():
        for inputs, targets in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            outputs.append(model(inputs.to(device)).cpu())
            labels.append(targets)
    logits, targets = torch.cat(outputs), torch.cat(labels)
    loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    competing = logits.clone()
    competing.scatter_(1, targets[:, None], -torch.inf)
    result = {'loss': loss.numpy(),
              'margin': (logits.gather(1, targets[:, None]).squeeze(1) -
                         competing.max(1).values).numpy(),
              'correct': (logits.argmax(1) == targets).numpy()}
    if return_logits:
        result.update(logits=logits.numpy(), labels=targets.numpy())
    return result


def _metrics(model, train, test, target_indices, config, device, prediction_path=None):
    predictions = _predict(model, test, device, config['eval_batch_size'],
                           return_logits=prediction_path is not None)
    fine, coarse = np.asarray(test.fine_targets), np.asarray(test.targets)
    target = config['target_fine_class']
    superclass = int(train.fine_to_coarse[target])

    def summarize(mask):
        return {'count': int(mask.sum()), **{
            key if key != 'correct' else 'accuracy': float(predictions[key][mask].mean())
            for key in ('loss', 'margin', 'correct')}}

    result = {
        'target': summarize(fine == target),
        'siblings': summarize((coarse == superclass) & (fine != target)),
        'other_superclasses': summarize(coarse != superclass),
        'overall': summarize(np.ones(len(test), dtype=bool)),
        'per_fine_class': {str(value): summarize(fine == value)
                           for value in sorted(set(fine.tolist()))},
        'per_superclass': {str(value): summarize(coarse == value)
                           for value in sorted(set(coarse.tolist()))},
    }
    member_losses = _predict(model, Subset(train, target_indices), device,
                             config['eval_batch_size'])['loss']
    nonmember_losses = predictions['loss'][fine == target]
    if prediction_path is not None:
        np.savez_compressed(prediction_path, **predictions,
                            test_index=np.arange(len(test)), fine_labels=fine,
                            member_index=np.asarray(target_indices), member_loss=member_losses)
    result['membership'] = {
        'loss_auc': float(roc_auc_score(
            np.r_[np.ones(len(member_losses)), np.zeros(len(nonmember_losses))],
            -np.r_[member_losses, nonmember_losses])),
        'members': len(member_losses), 'nonmembers': len(nonmember_losses),
        'member_loss_mean': float(member_losses.mean()),
        'nonmember_loss_mean': float(nonmember_losses.mean()),
        'interpretation': 'Diagnostic only: training fine-class records versus untouched same-fine test records.',
    }
    return result


def select_global_parameters(scores, percentage):
    """Rank scalar coordinates globally; controls subsequently match module counts."""
    if not math.isfinite(percentage) or not 0 < percentage <= 100:
        raise ValueError('parameter budget must be in (0, 100]')
    flat = torch.cat([value.detach().reshape(-1).cpu() for value in scores.values()])
    if not torch.isfinite(flat).all() or (flat < 0).any():
        raise ValueError('scores must be finite and nonnegative')
    count = int(flat.numel() * percentage / 100)
    if not count:
        raise ValueError('parameter budget selects no coordinates')
    chosen = torch.argsort(flat, descending=True, stable=True)[:count]
    selected, offset = {}, 0
    for name, value in scores.items():
        mask = torch.zeros(value.numel(), dtype=torch.bool)
        indices = chosen[(chosen >= offset) & (chosen < offset + value.numel())] - offset
        mask[indices] = True
        selected[name] = mask.reshape(value.shape).nonzero()
        offset += value.numel()
    return selected


def margin_gap_closure(original, candidate, gold, epsilon=1e-6):
    denominator = original - gold
    return None if abs(denominator) <= epsilon else (original - candidate) / denominator


def mask_bits(mask, shapes):
    """Boolean selection of every scalar, concatenated in parameter order."""
    bits = []
    for name, shape in shapes.items():
        flat = torch.zeros(math.prod(shape), dtype=torch.bool)
        coordinates = mask.get(name)
        if coordinates is not None and len(coordinates):
            # _flat_indices validates bounds and duplicates; it only reads the shape.
            flat[_flat_indices(torch.empty(shape, dtype=torch.bool),
                               torch.as_tensor(coordinates).cpu())] = True
        bits.append(flat)
    return torch.cat(bits).numpy()


def save_masks(path, masks, shapes):
    """Bit-packed masks: about 0.5 MB each compressed, and they outlive checkpoints."""
    names = list(shapes)
    arrays = {stem: np.packbits(mask_bits(mask, shapes)) for stem, mask in masks.items()}
    if {'parameter_names', 'parameter_numels'} & set(arrays):
        raise ValueError('mask stem collides with metadata key')
    np.savez_compressed(path, parameter_names=np.asarray(names),
                        parameter_numels=np.asarray([math.prod(shapes[n]) for n in names]),
                        **arrays)


def load_mask_bits(path, stem):
    with np.load(path) as stored:
        total = int(stored['parameter_numels'].sum())
        return np.unpackbits(stored[stem], count=total).astype(bool)


def summarize_scores(scores, top_fraction=0.1):
    """Per-module score mass and pairwise ranking agreement; small enough to keep."""
    flat = {name: torch.cat([v.detach().reshape(-1).float().cpu() for v in value.values()])
            for name, value in scores.items()}
    summary = {'top_fraction': top_fraction, 'total_mass': {},
               'per_module_mass_fraction': {}, 'agreement': []}
    for name, value in scores.items():
        total = float(flat[name].double().sum())
        modules = {}
        for parameter, tensor in value.items():
            module = parameter.rpartition('.')[0]
            modules[module] = modules.get(module, 0.) + float(tensor.double().sum())
        summary['total_mass'][name] = total
        summary['per_module_mass_fraction'][name] = {
            module: mass / total if total > 0 else None for module, mass in modules.items()}
    for first, second in itertools.combinations(flat, 2):
        summary['agreement'].append({'first': first, 'second': second,
                                     **score_agreement(flat[first], flat[second], top_fraction)})
    return summary


def gradient_norms(model, train, groups, config, device):
    """Mean-loss gradient norms at the trained point, eval-mode BatchNorm (paper (A3))."""
    model.to(device).eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    sums, counts = {}, {}
    for name, indices in groups.items():
        total = [torch.zeros_like(p) for p in parameters]
        for inputs, targets in DataLoader(Subset(train, indices), shuffle=False,
                                          batch_size=config['eval_batch_size']):
            loss = nn.functional.cross_entropy(model(inputs.to(device)), targets.to(device),
                                               reduction='sum')
            for accumulated, gradient in zip(total, torch.autograd.grad(loss, parameters)):
                accumulated.add_(gradient)
        sums[name], counts[name] = total, len(indices)
    norm = lambda tensors, count: float(torch.sqrt(sum((t / count).square().sum() for t in tensors)))
    result = {f'{name}_gradient_norm': norm(sums[name], counts[name]) for name in groups}
    if {'target', 'retained'} <= set(groups):
        both = [a + b for a, b in zip(sums['target'], sums['retained'])]
        result['full_gradient_norm'] = norm(both, counts['target'] + counts['retained'])
    result['batchnorm'] = 'eval mode (running statistics)'
    return result


def _named_like(flat, model):
    result, offset = {}, 0
    for name, parameter in model.named_parameters():
        size = parameter.numel()
        result[name] = flat[offset:offset + size].reshape(parameter.shape)
        offset += size
    return result


def _score(model, train, target_indices, swap_indices, config, seed, device,
           sample_indices=None, cross_indices=None):
    generator = torch.Generator().manual_seed(seed)

    def sample(indices, cap, source=generator):
        order = torch.randperm(len(indices), generator=source)[:cap].tolist()
        return [int(indices[index]) for index in order]

    samples = dict(sample_indices) if sample_indices is not None else {
        'full': sample(list(range(len(train))), config.get('spectral_max_samples', 2048)),
        'target': sample(target_indices, config.get('spectral_target_max_samples', 512)),
        'swap': sample(swap_indices, config.get('spectral_target_max_samples', 512)),
    }
    if cross_indices is not None and 'cross' not in samples:
        # Separate stream: full/target/swap draws stay identical to earlier runs.
        samples['cross'] = sample(cross_indices, config.get('spectral_target_max_samples', 512),
                                  torch.Generator().manual_seed(seed + 1))
    loaders = {name: DataLoader(Subset(train, indices), shuffle=False,
                               batch_size=config.get('spectral_batch_size', 32))
               for name, indices in samples.items()}
    _seed(seed)
    started = time.perf_counter()
    thresholds = dict(eigenvalue_threshold=config.get('spectral_eigenvalue_min', 1e-6),
                      relative_eigenvalue_threshold=config.get('spectral_eigenvalue_rtol', 1e-5))
    score = estimate_core_score(
        model.to(device), loaders['full'], loaders['target'], nn.CrossEntropyLoss(),
        config.get('spectral_rank', 20), device,
        num_power_iters=config.get('spectral_num_power_iters', 50),
        hmp_chunk_size=config.get('spectral_hmp_chunk_size', 1),
        curvature_backend='ggn', return_evecs=True, loader_control=loaders['swap'],
        power_tolerance=config.get('spectral_power_tolerance', 1e-3), **thresholds,
    )
    basis = score.pop('evecs')
    if not basis.shape[1]:
        raise RuntimeError('No positive curvature directions; reset rankings are undefined')
    swap_whitened = score['diagnostics']['control']['whitened_target']
    swap, leverage = torch.empty(basis.shape[0]), torch.empty(basis.shape[0])
    for start in range(0, basis.shape[0], 65536):
        rows = basis[start:start + 65536]
        swap[start:start + len(rows)] = (rows @ swap_whitened).square().sum(1)
        leverage[start:start + len(rows)] = rows.square().sum(1)
    scores = {'core': score['diag_by_name'], 'swap': _named_like(swap, model),
              'leverage': _named_like(leverage, model)}
    diagnostics = score['diagnostics']
    subspace_seconds = diagnostics['subspace']['elapsed_seconds']
    control_seconds = diagnostics['control']['elapsed_seconds']
    timings = {'subspace_seconds': subspace_seconds, 'swap_control_seconds': control_seconds,
               # Target products plus score assembly: the per-request part.
               'target_request_seconds': diagnostics['elapsed_seconds'] - subspace_seconds - control_seconds}
    if 'cross' in loaders:
        cross_started = time.perf_counter()
        matmat, _, _, _ = make_ggn_matvec(model, loaders['cross'], nn.CrossEntropyLoss(), device,
                                          hmp_chunk_size=config.get('spectral_hmp_chunk_size', 1))
        device_basis = basis.to(device)
        projection = (device_basis.T @ matmat(device_basis)).cpu()
        del device_basis
        cross = core_score_from_subspace(basis, score['evals_H'], projection, **thresholds)
        scores['cross'] = _named_like(cross['diag_flat'], model)
        diagnostics['cross_control'] = {
            'projected_target': cross['projected_target'], 'whitened_target': cross['whitened_target'],
            'core_trace': cross['diagnostics']['core_trace'],
            **score_agreement(score['diag_flat'], cross['diag_flat']),
            'elapsed_seconds': time.perf_counter() - cross_started}
        timings['cross_control_seconds'] = diagnostics['cross_control']['elapsed_seconds']
    timings['score_seconds'] = time.perf_counter() - started
    return scores, {'sample_indices': samples, 'evals_H': score['evals_H'],
                    'projected_target': score['projected_target'],
                    'diagnostics': diagnostics, 'timings': timings}


def _audit(recovered, original, shapes, bits):
    """Prove only selected scalars moved; BatchNorm buffers may follow recovery."""
    changed = torch.cat([(recovered[name] != original[name]).reshape(-1) for name in shapes]).numpy()
    unselected = int((changed & ~bits).sum())
    if unselected:
        raise RuntimeError(f'{unselected} unselected parameters changed during recovery')
    return {'selected_count': int(bits.sum()), 'unselected_parameter_changes': 0,
            'changed_selected_coordinates': int((changed & bits).sum()),
            'changed_buffer_tensors': sum(int(not torch.equal(recovered[name], original[name]))
                                          for name in original if name not in shapes)}


def _validate(config, device):
    if config['dataset_name'] != 'cifar100_superclass' or config['num_classes'] != 20:
        raise ValueError('within-class unlearning requires CIFAR-100 with 20 superclass outputs')
    if config['model_name'] != 'resnet18':
        raise ValueError('within-class unlearning requires ResNet-18')
    if (config['distribution_type'], config['num_clients'], config['target_client']) != ('fine_class', 6, 0):
        raise ValueError('expected fine_class partition with target client 0 and five retained clients')
    if (isinstance(config['num_gold_models'], bool) or not isinstance(config['num_gold_models'], int)
            or config['num_gold_models'] < 3):
        raise ValueError('at least three gold references are required')
    if config.get('spectral_curvature_backend', 'ggn') != 'ggn':
        raise ValueError('within-class scoring requires positive-semidefinite GGN curvature')
    if config.get('drop_last', False):
        raise ValueError('drop_last must be false to train on every retained record')
    for name in ('target_fine_class', 'swap_fine_class', 'cross_fine_class'):
        if name == 'cross_fine_class' and name not in config:
            continue
        if isinstance(config[name], bool) or not isinstance(config[name], int) or not 0 <= config[name] < 100:
            raise ValueError(f'{name} must be an integer in [0, 99]')
    for name in ('train_epochs', 'recovery_epochs', 'batch_size', 'eval_batch_size'):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not config['parameter_budgets'] or len(set(config['parameter_budgets'])) != len(config['parameter_budgets']):
        raise ValueError('parameter budgets must be nonempty and distinct')
    if any(not math.isfinite(value) or not 0 < value <= 100 for value in config['parameter_budgets']):
        raise ValueError('parameter budgets must be in (0, 100]')
    decay = config.get('weight_decay', 0.0)
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or decay < 0:
        raise ValueError('weight_decay must be a finite nonnegative number')
    if not isinstance(config.get('augment', False), bool):
        raise ValueError('augment must be a boolean')
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu only for local checks')


def _execute(config, output, seed, device, train, test, model_factory):
    config = {**config, 'save_models': config.get('save_models', False)}
    _validate(config, device)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'predictions').mkdir()
    save_models = config.get('save_models', False)
    started = time.perf_counter()
    _seed(seed)
    clients = get_clients_subsets(train, config)
    target_indices = list(map(int, clients[0].indices))
    retained_indices = sorted(int(index) for client in clients[1:] for index in client.indices)
    fine = np.asarray(train.fine_targets)
    target, swap = config['target_fine_class'], config['swap_fine_class']
    cross = config.get('cross_fine_class')
    if target == swap or train.fine_to_coarse[target] != train.fine_to_coarse[swap]:
        raise ValueError('swap must be a different retained fine class in the target superclass')
    if cross is not None and train.fine_to_coarse[cross] == train.fine_to_coarse[target]:
        raise ValueError('cross control must come from a different superclass than the target')
    swap_indices = np.flatnonzero(fine == swap).tolist()
    cross_indices = np.flatnonzero(fine == cross).tolist() if cross is not None else None
    if (set(target_indices) != set(np.flatnonzero(fine == target).tolist()) or
            len(set(target_indices + retained_indices)) != len(train) or
            len(target_indices) + len(retained_indices) != len(train)):
        raise ValueError('clients must partition all training records, with the complete fine class isolated')
    for name, indices in (('swap', swap_indices), ('cross', cross_indices)):
        if indices is not None and (not indices or not set(indices).issubset(retained_indices)):
            raise ValueError(f'{name} fine-class records must all be retained')
    gold_seeds = [seed * 1000 + 100 + index for index in range(config['num_gold_models'])]
    seeds = {'original_initialization': seed, 'original_training': seed,
             'gold_initialization': [seed, *[seed * 1000 + 200 + index
                                             for index in range(1, config['num_gold_models'])]],
             'gold_training': gold_seeds,
             'curvature': seed * 1000 + 300 + config.get('spectral_seed', 0),
             'recovery': seed * 1000 + 400, 'random_selection': seed * 1000 + 500}
    protocol = {'config': config, 'seed': seed, 'seeds': seeds,
                'dataset_sizes': {'train': len(train), 'test': len(test)},
                'fine_classes': train.fine_classes, 'coarse_classes': train.classes,
                'fine_to_coarse': train.fine_to_coarse,
                'target_fine_name': train.fine_classes[target],
                'swap_fine_name': train.fine_classes[swap],
                'cross_fine_name': train.fine_classes[cross] if cross is not None else None,
                'target_superclass': int(train.fine_to_coarse[target]),
                'target_indices': target_indices, 'retained_indices': retained_indices,
                'swap_indices': swap_indices, 'cross_indices': cross_indices,
                'client_indices': [list(map(int, client.indices)) for client in clients],
                'reset': 'original pretraining initialization',
                'selection': 'global scalar count; swap, cross, leverage and random match core counts per module',
                'recovery': 'selected coordinates only; original BatchNorm buffers copied to every arm and updated on retained data',
                'training_recipe': ('augmented (32-px crop, pad 4, flip) with weight decay %g' % config.get('weight_decay', 0.0)
                                    if config.get('augment') else 'no augmentation, weight decay %g' % config.get('weight_decay', 0.0)),
                'test': 'untouched official test split; no training holdout',
                'gold_reference': 'first shares original initialization; other gold initializations and all gold training seeds independent',
                'membership': 'negative-loss AUC; target training members against same-fine test nonmembers; diagnostic only'}
    _write_json(output / 'protocol.json', protocol)
    fit = training_view(train, config)
    _seed(seed)
    original = model_factory().to(device)
    initial = _state(original)
    if save_models:
        torch.save(initial, output / 'initial.pt')
    logging.info('Training original model on %s records', len(train))
    training_started = time.perf_counter()
    _train(original, fit, config['train_epochs'], config, device, seed)
    timings = {'original_training_seconds': time.perf_counter() - training_started}
    original_state = _state(original)
    shapes = {name: tuple(parameter.shape) for name, parameter in original.named_parameters()}
    if save_models:
        torch.save(original_state, output / 'original.pt')
    gradient_started = time.perf_counter()
    stationarity = gradient_norms(original, train, {'target': target_indices, 'retained': retained_indices},
                                  config, device)
    timings['stationarity_seconds'] = time.perf_counter() - gradient_started
    evaluate = lambda model, name: _metrics(
        model, train, test, target_indices, config, device, output / 'predictions' / f'{name}.npz')
    results = {'seed': seed, 'original': evaluate(original, 'original'), 'golds': [], 'arms': [],
               'stationarity': stationarity, 'stage_timings': timings}
    retained = Subset(fit, retained_indices)
    for index, gold_seed in enumerate(gold_seeds):
        logging.info('Training gold reference %s/%s on %s retained records',
                     index + 1, len(gold_seeds), len(retained))
        _seed(seeds['gold_initialization'][index])
        gold = model_factory().to(device)
        if index == 0:
            gold.load_state_dict(initial)
        training_started = time.perf_counter()
        _train(gold, retained, config['train_epochs'], config, device, gold_seed)
        training_seconds = time.perf_counter() - training_started
        if save_models:
            torch.save(_state(gold), output / f'gold_{index}.pt')
        results['golds'].append({'initialization_seed': seeds['gold_initialization'][index],
                                  'training_seed': gold_seed, 'training_seconds': training_seconds,
                                  'metrics': evaluate(gold, f'gold_{index}')})
        del gold
    logging.info('Estimating target core, same-superclass swap, cross-superclass and leverage scores')
    scores, diagnostics = _score(original, train, target_indices, swap_indices,
                                  config, seeds['curvature'], device, cross_indices=cross_indices)
    timings['score'] = diagnostics.pop('timings')
    if save_models:
        torch.save({'scores': scores, **diagnostics}, output / 'scores.pt')
    _write_json(output / 'score_diagnostics.json', diagnostics)
    _write_json(output / 'score_summary.json', summarize_scores(scores))
    groups = ('target', 'siblings', 'other_superclasses', 'overall')
    gold_summary = {group: {metric: {
        'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1)),
        'min': min(values), 'max': max(values)}
        for metric in ('accuracy', 'loss', 'margin')
        for values in [[gold['metrics'][group][metric] for gold in results['golds']]]}
        for group in groups}
    results['gold_reference_summary'] = gold_summary
    total = sum(value.numel() for value in scores['core'].values())
    packed = {}
    for budget_index, percentage in enumerate(config['parameter_budgets']):
        selected = select_global_parameters(scores['core'], percentage)
        _seed(seeds['random_selection'] + budget_index)
        masks = {'core': selected,
                 'swap': find_informative_params_matching(scores['swap'], selected)}
        if 'cross' in scores:
            masks['cross'] = find_informative_params_matching(scores['cross'], selected)
        masks.update(leverage=find_informative_params_matching(scores['leverage'], selected),
                     random=find_random_params_matching(scores['core'], selected))
        for arm, mask in masks.items():
            logging.info('Recovering %s arm at %.3g%% scalar parameter budget', arm, percentage)
            recovery_started = time.perf_counter()
            model = UnlearnNet(original, mask, reset_reference=initial).to(device)
            _train(model, retained, config['recovery_epochs'], config, device, seeds['recovery'])
            recovered = copy.deepcopy(original)
            recovered.load_state_dict(model.get_retrained_params())
            recovery_seconds = time.perf_counter() - recovery_started
            stem = f'{arm}_{percentage:g}'
            bits = mask_bits(mask, shapes)
            audit = _audit(_state(recovered), original_state, shapes, bits)
            packed[stem] = mask
            metrics = evaluate(recovered, stem)
            reference_margins = [gold['metrics']['target']['margin'] for gold in results['golds']]
            closure = lambda reference: margin_gap_closure(
                results['original']['target']['margin'], metrics['target']['margin'], reference,
                config.get('margin_gap_epsilon', 1e-6))
            result = {'arm': arm, 'parameter_budget_percent': percentage,
                      'selected_count': sum(len(value) for value in mask.values()),
                      'total_parameters': total,
                      'recovery_seconds': recovery_seconds, 'audit': audit,
                      'selected_per_module': {}, 'metrics': metrics,
                      'target_margin_gap_closure': closure(float(np.mean(reference_margins))),
                      'target_margin_gap_closure_per_gold': [closure(value) for value in reference_margins],
                      'accuracy_delta_from_gold_mean': {
                          group: metrics[group]['accuracy'] - gold_summary[group]['accuracy']['mean']
                          for group in groups}}
            for name, coordinates in mask.items():
                module = name.rpartition('.')[0]
                result['selected_per_module'][module] = result['selected_per_module'].get(module, 0) + len(coordinates)
            if save_models:
                torch.save(mask, output / f'{stem}_mask.pt')
                torch.save(_state(recovered), output / f'{stem}.pt')
            results['arms'].append(result)
            del model, recovered
    save_masks(output / 'masks.npz', packed, shapes)
    timings['total_seconds'] = time.perf_counter() - started
    _write_json(output / 'results.json', results)
    return results


def run(config, output: Path, seed: int, device='cuda'):
    """Run one original and three or more gold references without overwriting a run."""
    config, output = dict(config), Path(output)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    _validate(config, device)
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose a new output directory')
    train, test = get_datasets(config)
    return _execute(config, output, seed, device, train, test, lambda: create_resnet(config))


def synthetic_datasets(config):
    """Tiny offline stand-in with CIFAR-100's fine/superclass metadata."""
    mapping = list(np.repeat(np.arange(20), 5).astype(int))
    target, swap = config['target_fine_class'], config['swap_fine_class']
    sibling = next(value for value in range(100) if value != target and mapping[value] == mapping[target])
    mapping[sibling], mapping[swap] = mapping[swap], mapping[sibling]
    generator = torch.Generator().manual_seed(2026)

    def dataset(repeats):
        fine = np.repeat(np.arange(100), repeats).tolist()
        data = TensorDataset(torch.randn(len(fine), 4, generator=generator),
                             torch.tensor([mapping[value] for value in fine]))
        data.targets = [mapping[value] for value in fine]
        data.fine_targets, data.fine_to_coarse = fine, mapping
        data.fine_classes, data.classes = [f'fine_{value}' for value in range(100)], [f'coarse_{value}' for value in range(20)]
        return data

    return dataset(2), dataset(1)


SMOKE_OVERRIDES = {'train_epochs': 1, 'recovery_epochs': 1, 'num_gold_models': 3,
                   'batch_size': 64, 'eval_batch_size': 256, 'parameter_budgets': [8.0],
                   'spectral_rank': 2, 'spectral_max_samples': 8,
                   'spectral_target_max_samples': 2, 'spectral_num_power_iters': 1,
                   'spectral_batch_size': 4, 'augment': False}


def smoke_model():
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 20))


def smoke(config, device='cpu'):
    """Exercise training, shared-basis swap, matched resets, recovery and metrics offline."""
    config = {**config, **SMOKE_OVERRIDES}
    _seed(2026)
    train, test = synthetic_datasets(config)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='within-class-smoke-') as directory:
            result = _execute(config, Path(directory) / 'run', 2026, device, train, test, smoke_model)
            arms = 5 if config.get('cross_fine_class') is not None else 4
            assert len(result['golds']) == 3 and len(result['arms']) == arms
            assert len({arm['selected_count'] for arm in result['arms']}) == 1
            assert all(arm['audit']['unselected_parameter_changes'] == 0 for arm in result['arms'])
    finally:
        torch.set_num_threads(threads)
    print(f'PASS: within-class train, GGN score, swap/cross, matched recovery and evaluation on {device}')
    return result
