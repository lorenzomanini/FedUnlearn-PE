"""Synthetic saved-output regression check; no training or measured results.

Run with .venv/bin/python validation/check_classical_audit.py.
"""
import csv
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(17)


def dump(directory, name, value):
    with (directory / name).open('wb') as stream:
        pickle.dump(value, stream)


with tempfile.TemporaryDirectory(prefix='classical-audit-') as temporary:
    root = Path(temporary)
    source, output = root / 'suite', root / 'audit'
    directory = source / 'test_0'
    directory.mkdir(parents=True)
    labels = np.repeat(np.arange(10), 4)
    clients = [list(range(4*c, 4*c+4)) for c in range(10)]
    clients[-1].pop()  # One record never assigned; others may be held out.
    target = np.array(clients[5])
    training = np.array([i for c, indices in enumerate(clients)
                         for i in (indices if c == 5 else indices[:-1])])
    cases = [dict(unlearning_percentage=0, retrain_epochs=1, tests=['LiRA']),
             dict(unlearning_percentage=10, retrain_epochs=1, tests=['LiRA'])]
    config = dict(distribution_type='categorical', target_client=5, num_tests=1,
                  save_models='baseline', lira_global_variance=True)
    dump(source, 'labels.pkl', {'train': labels, 'test': labels})
    dump(source, 'clients_indices.pkl', clients)
    dump(source, 'init_params.pkl', config)
    dump(source, 'test_params.pkl', cases)
    dump(source, 'stage_timings.pkl', {'total_suite_seconds': 1})
    membership = np.array([True]*4 + [False]*40)
    candidates = np.concatenate((target, np.arange(40) + 40))
    shadow_membership = np.tile(np.array([True, True, False, False])[:, None], (1, 44))
    np.savez(source / 'lira_shadow_bank.npz', candidate_membership=membership,
             candidate_complete_indices=candidates, scores=rng.normal(size=(4, 44)),
             shadow_membership=shadow_membership)
    diagnostics = dict(training_indices=training, information_method='diagonal',
                       hessian_method='diag_ggn_mc', stochastic_correction=False,
                       diagonal_population='training', full_samples=len(training), target_samples=4)
    dump(directory, 'score_diagnostics.pkl', diagnostics)
    extra = {'case_index': [0, 1], 'reset_params_percentage': [0, 1]}
    for key in ('selection_seconds', 'recovery_seconds', 'unlearning_with_score_seconds',
                'random_baseline_seconds', 'evaluation_seconds', 'lira_evaluation_seconds'):
        extra[key] = [1, 1]
    dump(directory, 'extra_results.pkl', extra)
    dump(directory, 'stage_timings.pkl', {'score_seconds': 1, 'gold_retraining_seconds': 2})
    prediction = {'pred': labels.copy(), 'loss': np.ones(40)}
    initial = {'trained': prediction, 'shadow_out': prediction}
    for split in ('train', 'test'):
        dump(directory, f'initial_eval_{split}_results.pkl', initial)
        arrays = {f'{arm}__{field}': np.stack([prediction[field]] * 2)
                  for arm in ('reset', 'retrained', 'random_reset', 'random_retrained')
                  for field in ('pred', 'loss')}
        np.savez(directory / f'eval_{split}_results.npz', **arrays)
    lira = rng.normal(size=44)
    dump(directory, 'initial_lira_results.pkl', {'trained': lira, 'shadow_out': lira})
    dump(directory, 'lira_case_indices.pkl', [0, 1])
    np.savez(directory / 'eval_lira_results.npz', **{
        arm: np.stack([lira] * 2) for arm in ('reset', 'retrained', 'random_reset', 'random_retrained')})
    for name in ('initial', 'original', 'gold'):
        (directory / f'{name}_model.pth').touch()
    command = [sys.executable, 'validation/accuracy_audit/analyze_saved_cifar.py',
               '--input', str(source), '--output', str(output), '--allow-inefficient']
    for distribution in ('categorical', 'random', 'dirichlet'):
        config['distribution_type'] = distribution
        dump(source, 'init_params.pkl', config)
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        metadata = json.loads((output / 'cifar_audit_metadata.json').read_text())
        assert metadata['split_sizes']['unassigned'] == 1
        assert metadata['split_sizes']['assigned_retained_heldout'] == 9
        rows = list(csv.DictReader((output / 'cifar_per_class_per_run.csv').open()))
        assert len(rows) == 100 and {int(row['class_label']) for row in rows} == set(range(10))
        utility = next(csv.DictReader((output / 'cifar_utility_per_run.csv').open()))
        assert float(utility['assigned_retain_heldout_accuracy_pct']) == 100
        assert float(utility['unassigned_accuracy_pct']) == 100
        if distribution == 'categorical':
            assert float(utility['retained_class_test_accuracy_pct']) == 100
    extra['case_index'] = [0]  # Swallowed runner failures must make the audit fail.
    dump(directory, 'extra_results.pkl', extra)
    assert subprocess.run(command, cwd=ROOT, capture_output=True).returncode != 0

print('PASS: categorical/random/Dirichlet audit, separate unassigned records, incomplete-case failure')
