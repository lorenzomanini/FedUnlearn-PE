"""Re-score saved TMLR repetitions with paired arms, without retraining.

Requires initialization, original and gold weights from --save-models or a temporary
--checkpoint-dir, plus the saved training/target indices (`score_diagnostics.pkl`).
This script reloads them and evaluates, on the *same* original model:

  core       the rank-r core score (recomputed with the suite's spectral settings)
  leverage   [U U^T]_ii, the target-independent part of the core allocation
  ratio      core / leverage, the leverage-normalized allocation
  contrast   [U (B - I)_+^2 U^T]_ii, the core with the isotropic part removed
  diag_exact (h_T / h)^2 from exact GGN diagonals on all training records
  diag_mc    the TMLR diagonal arm (Monte-Carlo GGN diagonal)

plus a layer-matched random reset per arm, fine-tuning all weights without a reset,
and resetting the forget class's row of the output layer. Each model is measured
against the gold model: per-class test accuracy, agreement and Jensen-Shannon
divergence on retained-class test inputs, and a linear probe for the forget class on
frozen penultimate features. Because every arm shares one original model, arm
differences can be tested pairwise across repetitions.

Usage (inside a GPU job):
  python -m experiments.rescore_saved --suite stat_tests/TMLR/<suite>_job_<id>/<suite> \
      --forget-class 5 --reps 0-19 --output rescore/<suite>
"""
import argparse
import copy
import json
import os
import pickle
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from experiments import runner
from experiments.datasets import get_datasets
from experiments.models import get_loss_class, get_model_class
from fisherunlearn import compute_client_information
from fisherunlearn.information.selection import find_informative_params, find_random_params_matching
from fisherunlearn.information.spectral_wip import estimate_core_score
from fisherunlearn.unlearning import UnlearnNet, reset_parameters

MASSES = (2.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0)


def select(scores, mass, scope):
    """Per-layer prefix (the method's rule) or one global prefix over all weights."""
    if scope == 'layer':
        return find_informative_params(scores, 'information', mass)
    names = list(scores)
    flat = torch.cat([scores[n].reshape(-1) for n in names])
    chosen = find_informative_params({'global.flat': flat}, 'information', mass)['global.flat'].reshape(-1)
    out, offset = {}, 0
    for n in names:
        size = scores[n].numel()
        local = chosen[(chosen >= offset) & (chosen < offset + size)] - offset
        mask = torch.zeros(size, dtype=torch.bool)
        mask[local] = True
        out[n] = torch.argwhere(mask.reshape(scores[n].shape))
        offset += size
    return out


def flat_named(model, flat):
    out, offset = {}, 0
    for name, p in model.named_parameters():
        out[name] = flat[offset:offset + p.numel()].reshape(p.shape).float().cpu()
        offset += p.numel()
    return out


def arms_from_core(score):
    U = score['evecs'].double()
    B = score['whitened_target'].double()
    B = (B + B.T) / 2
    core = ((U @ B) ** 2).sum(1)
    leverage = (U ** 2).sum(1)
    ok = leverage > 1e-12 * leverage.max()
    ratio = torch.zeros_like(core)
    ratio[ok] = core[ok] / leverage[ok]
    mu, V = torch.linalg.eigh(B)
    contrast = ((U @ (V @ torch.diag((mu - 1).clamp_min(0)) @ V.T)) ** 2).sum(1)
    return {'core': core, 'leverage': leverage, 'ratio': ratio, 'contrast': contrast,
            'isotropic_share': B.shape[0] / float((mu ** 2).sum())}


@torch.no_grad()
def outputs_and_features(model, dataset, device, batch_size=512):
    model.to(device).eval()
    last = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    feats = []
    handle = last.register_forward_hook(lambda mod, inp, out: feats.append(inp[0].detach().cpu()))
    logits, labels = [], []
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    handle.remove()
    model.cpu()
    return torch.cat(logits), torch.cat(labels), torch.cat(feats)


def measure(model, test, device, forget_class, gold_cache):
    logits, labels, feats = outputs_and_features(model, test, device)
    pred = logits.argmax(1)
    forget = labels == forget_class
    gold_logits, gold_prob = gold_cache
    prob = torch.softmax(logits, 1)
    mid = (prob + gold_prob) / 2
    js = 0.5 * ((prob * (prob.clamp_min(1e-12).log() - mid.log())).sum(1)
                + (gold_prob * (gold_prob.clamp_min(1e-12).log() - mid.log())).sum(1))
    half = len(labels) // 2
    target = forget.numpy().astype(int)
    order = np.random.default_rng(0).permutation(len(labels))
    a, b = order[:half], order[half:]
    probe = LogisticRegression(max_iter=3000).fit(feats[a].numpy(), target[a])
    per_class = [float((pred[labels == c] == c).float().mean()) * 100 for c in range(int(labels.max()) + 1)]
    return {
        'test': float((pred == labels).float().mean()) * 100,
        'forget': float((pred[forget] == labels[forget]).float().mean()) * 100,
        'retain': float((pred[~forget] == labels[~forget]).float().mean()) * 100,
        'per_class': per_class,
        'agree_gold_retain': float((pred == gold_logits.argmax(1))[~forget].float().mean()) * 100,
        'js_gold_retain': float(js[~forget].mean()),
        'probe_auc': float(roc_auc_score(target[b], probe.decision_function(feats[b].numpy()))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--suite', required=True)
    ap.add_argument('--checkpoint-dir', help='baseline checkpoint root; defaults to --suite')
    ap.add_argument('--forget-class', type=int, default=5)
    ap.add_argument('--reps', default='0-19')
    ap.add_argument('--arms', default='core,leverage,ratio,contrast,diag_exact,diag_mc')
    ap.add_argument('--random-for', default='core,diag_exact,ratio')
    ap.add_argument('--reset', default='initial', choices=['initial', 'zero'])
    ap.add_argument('--recovery-epochs', type=int, default=1)
    ap.add_argument('--selection', default='layer', choices=['layer', 'global'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    runner.set_device(device)
    runner.set_batch_sizes(128, 128, 128, 128)
    with open(os.path.join(args.suite, 'init_params.pkl'), 'rb') as f:
        config = pickle.load(f)
    train_dataset, test_dataset = get_datasets(dict(config))
    model_class = get_model_class(dict(config))
    loss_class = get_loss_class(dict(config))
    trainer = runner.get_trainer_function(dict(config))
    os.makedirs(args.output, exist_ok=True)
    lo, hi = (int(v) for v in args.reps.split('-'))
    for rep in range(lo, hi + 1):
        rep_dir = os.path.join(args.suite, f'test_{rep}')
        out_path = os.path.join(args.output, f'rep_{rep}.json')
        if os.path.exists(out_path):
            continue
        started = time.perf_counter()
        torch.manual_seed(config.get('repetition_seed', 0) + rep)
        np.random.seed(config.get('repetition_seed', 0) + rep)
        with open(os.path.join(rep_dir, 'score_diagnostics.pkl'), 'rb') as f:
            diag = pickle.load(f)
        training = list(diag['training_indices'])
        target = list(diag['target_indices'])
        target_set = set(target)
        retained = [i for i in training if i not in target_set]
        model_dir = os.path.join(args.checkpoint_dir or args.suite, f'test_{rep}')
        if not all(os.path.isfile(os.path.join(model_dir, name))
                   for name in ('initial_model.pth', 'original_model.pth', 'gold_model.pth')):
            ap.error(f'Missing checkpoints in {model_dir}; rerun the suite with --save-models '
                     'or point --checkpoint-dir to an existing checkpoint root.')
        initial = torch.load(os.path.join(model_dir, 'initial_model.pth'))
        original, gold = model_class(), model_class()
        original.load_state_dict(torch.load(os.path.join(model_dir, 'original_model.pth')))
        gold.load_state_dict(torch.load(os.path.join(model_dir, 'gold_model.pth')))
        gold_logits, _, _ = outputs_and_features(gold, test_dataset, device)
        gold_cache = (gold_logits, torch.softmax(gold_logits, 1))
        result = {'rep': rep, 'reference': {
            'original': measure(original, test_dataset, device, args.forget_class, gold_cache),
            'gold': measure(gold, test_dataset, device, args.forget_class, gold_cache)}}
        retained_ds, target_ds = Subset(train_dataset, retained), Subset(train_dataset, target)

        scores, timing = {}, {}
        needs_core = {'core', 'leverage', 'ratio', 'contrast'} & set(args.arms.split(','))
        if needs_core:
            t = time.perf_counter()
            seed = config.get('spectral_seed', 0)
            full = runner._curvature_loader(Subset(train_dataset, training), config.get('spectral_max_samples', 512),
                                            128, seed)
            tgt = runner._curvature_loader(target_ds, config.get('spectral_target_max_samples', 512), 128, seed + 1)
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
                torch.manual_seed(seed)
                score = estimate_core_score(
                    original.to(device), full, tgt, loss_class(), config.get('spectral_rank', 20), device,
                    num_power_iters=config.get('spectral_num_power_iters', 50),
                    power_tolerance=config.get('spectral_power_tolerance', 1e-3),
                    curvature_backend=config.get('spectral_curvature_backend', 'full'),
                    eigenvalue_threshold=config.get('spectral_eigenvalue_min', 1e-6),
                    relative_eigenvalue_threshold=config.get('spectral_eigenvalue_rtol', 1e-5),
                    return_evecs=True)
            original.cpu()
            derived = arms_from_core(score)
            result['isotropic_share'] = derived.pop('isotropic_share')
            for name, flat in derived.items():
                scores[name] = flat_named(original, flat)
            timing['core_seconds'] = time.perf_counter() - t
        for method, name in (('diag_ggn', 'diag_exact'), ('diag_ggn_mc', 'diag_mc')):
            if name in args.arms.split(','):
                t = time.perf_counter()
                # client 1 is the target; the total runs over all training records
                scores[name] = compute_client_information(
                    1, original, loss_class(), [retained_ds, target_ds], method=method,
                    learning_rate=config['learning_rate'], momentum=config['momentum'])
                timing[f'{name}_seconds'] = time.perf_counter() - t

        reset_reference = initial if args.reset == 'initial' else None
        rows = []
        for name in args.arms.split(','):
            for mass in MASSES:
                selected = select(scores[name], mass, args.selection)
                count = sum(len(v) for v in selected.values())
                if not count:
                    continue
                for kind in ('score', 'random'):
                    if kind == 'random' and name not in args.random_for.split(','):
                        continue
                    chosen = selected if kind == 'score' else find_random_params_matching(scores[name], selected)
                    base = model_class()
                    base.load_state_dict(reset_parameters(original, chosen, reset_reference=reset_reference))
                    net = UnlearnNet(base, chosen, reset_reference=reset_reference)
                    trainer(net, loss_class(), [retained_ds], [], args.recovery_epochs)
                    recovered = model_class()
                    recovered.load_state_dict(net.get_retrained_params())
                    rows.append({'arm': name, 'kind': kind, 'mass': mass,
                                 'params_pct': 100 * count / sum(p.numel() for p in original.parameters()),
                                 **measure(recovered, test_dataset, device, args.forget_class, gold_cache)})
        # baselines that do not use a score
        tuned = trainer(copy.deepcopy(original), loss_class(), [retained_ds], [], args.recovery_epochs)
        result['reference']['finetune_all'] = measure(tuned, test_dataset, device, args.forget_class, gold_cache)
        head = [m for m in original.named_modules() if isinstance(m[1], nn.Linear)][-1][0]
        width = dict(original.named_parameters())[f'{head}.weight'].shape[1]
        row_sel = {f'{head}.weight': torch.stack([torch.full((width,), args.forget_class), torch.arange(width)], 1),
                   f'{head}.bias': torch.tensor([[args.forget_class]])}
        base = model_class()
        base.load_state_dict(reset_parameters(original, row_sel, reset_reference=reset_reference))
        net = UnlearnNet(base, row_sel, reset_reference=reset_reference)
        trainer(net, loss_class(), [retained_ds], [], args.recovery_epochs)
        head_model = model_class()
        head_model.load_state_dict(net.get_retrained_params())
        result['reference']['output_row_reset'] = measure(head_model, test_dataset, device, args.forget_class, gold_cache)
        result.update(rows=rows, timing=timing, seconds=time.perf_counter() - started)
        with open(out_path, 'w') as f:
            json.dump(result, f)
        print(f'rep {rep}: {len(rows)} cases in {result["seconds"]:.0f} s', flush=True)


if __name__ == '__main__':
    main()
