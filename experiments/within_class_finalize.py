"""Audit, compact and optionally delete the checkpoints of a finished within-class seed.

Run after every controls job for the seed has completed. Before anything is
deleted it: (1) checks that each controls report is complete and points at this
experiment; (2) audits every saved arm (no unselected coordinate moved, selected
count, mask consistent with the saved score ranking); (3) writes bit-packed masks,
score summaries and gradient-norm diagnostics if the run predates them; and
(4) records a SHA-256 manifest of every deleted file. Any failure stops deletion.
"""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from experiments import within_class as wc
from experiments.datasets import get_datasets
from experiments.models import create_resnet


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_controls(experiment, controls):
    for directory in controls:
        report = json.loads((Path(directory) / 'results.json').read_text())
        if report.get('status') != 'complete':
            raise RuntimeError(f'{directory} is not complete (status {report.get("status")!r})')
        if Path(report['source']).resolve() != experiment.resolve():
            raise RuntimeError(f'{directory} was computed from {report["source"]}, not {experiment}')


def _ranking_consistent(score, bits, shapes, per_module):
    """Every selected score is >= every unselected score (globally or per module)."""
    flat = torch.cat([score[name].detach().reshape(-1).float() for name in shapes]).numpy()
    if not per_module:
        groups = {'global': np.ones(flat.size, dtype=bool)}
    else:
        groups, offset = {}, 0
        for name, shape in shapes.items():
            size = int(np.prod(shape))
            groups.setdefault(name.rpartition('.')[0], np.zeros(flat.size, dtype=bool))[offset:offset + size] = True
            offset += size
    for members in groups.values():
        chosen, rest = flat[members & bits], flat[members & ~bits]
        if chosen.size and rest.size and rest.max() > chosen.min() + 1e-12:
            return False
    return True


def audit(experiment, results, scores, original, shapes):
    rows, all_bits = [], {}
    for arm in results['arms']:
        name, budget = arm['arm'], arm['parameter_budget_percent']
        stem = f'{name}_{budget:g}'
        bits = all_bits[name, budget] = wc.mask_bits(_load(experiment / f'{stem}_mask.pt'), shapes)
        check = wc._audit(_load(experiment / f'{stem}.pt'), original, shapes, bits)
        if check['selected_count'] != arm['selected_count']:
            raise RuntimeError(f'{stem}: mask has {check["selected_count"]} coordinates, results say {arm["selected_count"]}')
        ranking = None
        if name in scores:
            ranking = _ranking_consistent(scores[name], bits, shapes, per_module=name != 'core')
            if not ranking:
                raise RuntimeError(f'{stem}: mask disagrees with the saved {name} score ranking')
        rows.append({'seed': results['seed'], 'arm': name, 'budget_percent': budget,
                     'actual_budget_percent': 100 * check['selected_count'] / bits.size,
                     'score_ranking_verified': ranking, **check})
    for row in rows:
        core = all_bits['core', row['budget_percent']]
        bits = all_bits[row['arm'], row['budget_percent']]
        shared = int((bits & core).sum())
        row.update(core_overlap_fraction=shared / int(bits.sum()),
                   core_jaccard=shared / int((bits | core).sum()))
    return rows


def compact(experiment, device):
    """Write everything the analysis needs that only checkpoints can provide."""
    protocol = json.loads((experiment / 'protocol.json').read_text())
    results = json.loads((experiment / 'results.json').read_text())
    original = _load(experiment / 'original.pt')
    saved = _load(experiment / 'scores.pt')
    config = {**protocol['config'], 'download': False}
    model = create_resnet(config)
    model.load_state_dict(original)
    shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    rows = audit(experiment, results, saved['scores'], original, shapes)
    with (experiment / 'checkpoint_audit.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    masks_path = experiment / 'masks.npz'
    if not masks_path.exists():
        stems = [f"{a['arm']}_{a['parameter_budget_percent']:g}" for a in results['arms']]
        masks = {stem: _load(experiment / f'{stem}_mask.pt') for stem in stems}
        wc.save_masks(masks_path, masks, shapes)
        for stem, mask in masks.items():
            if not np.array_equal(wc.load_mask_bits(masks_path, stem), wc.mask_bits(mask, shapes)):
                raise RuntimeError(f'Packed mask {stem} does not round-trip')
    if not (experiment / 'score_summary.json').exists():
        wc._write_json(experiment / 'score_summary.json', wc.summarize_scores(saved['scores']))
    if not (experiment / 'stationarity.json').exists() and 'stationarity' not in results:
        train, _ = get_datasets(config)
        stationarity = wc.gradient_norms(model, train, {'target': protocol['target_indices'],
                                                        'retained': protocol['retained_indices']},
                                         config, device)
        wc._write_json(experiment / 'stationarity.json', stationarity)


def keep_scores(experiment):
    """Lossless float32 score vectors (~41 MB per score). float16 underflows small scores."""
    path = experiment / 'scores.npz'
    if path.exists():
        return
    scores = _load(experiment / 'scores.pt')['scores']
    names = list(scores['core'])
    np.savez_compressed(path, parameter_names=np.asarray(names), **{
        score: torch.cat([value[name].reshape(-1).float() for name in names]).numpy()
        for score, value in scores.items()})


def delete_checkpoints(experiment):
    paths = sorted(experiment.glob('*.pt'))
    manifest = [{'file': path.name, 'bytes': path.stat().st_size, 'sha256': _sha256(path)}
                for path in paths]
    wc._write_json(experiment / 'deleted_checkpoints.json', manifest)
    for path in paths:
        path.unlink()
    logging.info('Deleted %s checkpoint files (%.2f GB)', len(paths),
                 sum(item['bytes'] for item in manifest) / 1e9)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path, help='seed_*/experiment directory')
    parser.add_argument('--controls', nargs='*', type=Path, default=[],
                        help='controls directories that must be complete before deletion')
    parser.add_argument('--delete-checkpoints', action='store_true')
    parser.add_argument('--keep-scores', action='store_true',
                        help='also keep full per-coordinate scores as scores.npz (~41 MB per score)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    experiment = args.experiment.resolve()
    verify_controls(experiment, args.controls)
    if not (experiment / 'original.pt').exists():
        raise FileNotFoundError(f'{experiment} has no checkpoints to audit')
    compact(experiment, args.device)
    if args.keep_scores:
        keep_scores(experiment)
    if args.delete_checkpoints:
        delete_checkpoints(experiment)
    print(f'PASS: {experiment} audited and compacted'
          + ('; checkpoints deleted' if args.delete_checkpoints else ''))


if __name__ == '__main__':
    main()
