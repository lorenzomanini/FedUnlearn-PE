"""The full-run summarizer must accept LiRA results for every case."""

import csv
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from validation.accuracy_audit import analyze_saved_cifar


class FullCifarSummaryTests(unittest.TestCase):
    def test_all_case_lira_and_efficiency_summary(self):
        def dump(path, value):
            with path.open('wb') as stream:
                pickle.dump(value, stream)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'suite'
            result = source / 'test_0'
            result.mkdir(parents=True)
            output = Path(temporary) / 'summary'
            candidates = np.array([0, 1, 2, 3, 12, 13, 14, 15])
            membership = np.array([True] * 4 + [False] * 4)
            shadows = np.array([[(row + col) % 2 == 0 for col in range(8)]
                                for row in range(8)])
            bank = {
                'scores': np.random.default_rng(7).normal(size=(8, 8)),
                'shadow_membership': shadows,
                'candidate_membership': membership,
                'candidate_complete_indices': candidates,
            }
            np.savez(source / 'lira_shadow_bank.npz', **bank)
            train_labels = np.zeros(12, dtype=int)
            train_labels[2:4] = 1
            dump(source / 'labels.pkl', {'train': train_labels,
                                         'test': np.array([0, 1, 0, 2])})
            dump(source / 'clients_indices.pkl', [list(range(4)), list(range(4, 12))])
            dump(source / 'init_params.pkl', {'num_tests': 1, 'target_client': 0,
                                              'distribution_type': 'preferential_class'})
            dump(source / 'test_params.pkl', [
                {'unlearning_percentage': mass, 'tests': ['LiRA']}
                for mass in (0.0, 50.0)
            ])
            dump(source / 'stage_timings.pkl', {'lira_shadow_training_seconds': 1.0})
            dump(result / 'score_diagnostics.pkl', {
                'training_indices': list(range(10)),
                'diagnostics': {
                    'retained_eigenpair_relative_residuals': [0.1],
                    'retained_rank': 1, 'subspace': {'converged': False},
                    'sum_rule_relative_error': 0.0,
                },
                'full_samples': 5, 'target_samples': 4,
            })
            dump(result / 'stage_timings.pkl', {'score_seconds': 1.0,
                                                'gold_retraining_seconds': 10.0})
            train = {'pred': np.zeros(12, dtype=int), 'loss': np.ones(12)}
            test = {'pred': np.zeros(4, dtype=int), 'loss': np.ones(4)}
            dump(result / 'initial_eval_train_results.pkl', {'trained': train, 'shadow_out': train})
            dump(result / 'initial_eval_test_results.pkl', {'trained': test, 'shadow_out': test})
            initial_scores = np.linspace(-1, 1, 8)
            dump(result / 'initial_lira_results.pkl', {
                'trained': initial_scores, 'shadow_out': initial_scores,
            })
            names = ('reset', 'retrained', 'random_reset', 'random_retrained')
            np.savez(result / 'eval_train_results.npz', **{
                f'{name}__{field}': np.stack([train[field], train[field]])
                for name in names for field in ('pred', 'loss')
            })
            np.savez(result / 'eval_test_results.npz', **{
                f'{name}__{field}': np.stack([test[field], test[field]])
                for name in names for field in ('pred', 'loss')
            })
            np.savez(result / 'eval_lira_results.npz', **{
                name: np.stack([initial_scores, initial_scores]) for name in names
            })
            dump(result / 'lira_case_indices.pkl', [0, 1])
            dump(result / 'extra_results.pkl', {
                'case_index': [0, 1], 'reset_params_percentage': [0.0, 1.0],
                'selection_seconds': [0.0, 0.1], 'recovery_seconds': [0.0, 0.5],
                'unlearning_with_score_seconds': [1.0, 1.6],
                'random_baseline_seconds': [0.0, 0.5],
                'evaluation_seconds': [0.1, 0.1],
                'lira_evaluation_seconds': [0.1, 0.1],
            })
            with mock.patch('sys.argv', [
                'audit', '--input', str(source), '--output', str(output),
            ]):
                analyze_saved_cifar.main()
            metadata = json.loads((output / 'cifar_audit_metadata.json').read_text())
            self.assertEqual(metadata['lira_case_indices'], [0, 1])
            self.assertTrue(metadata['efficiency_pass'])
            self.assertEqual(metadata['split_sizes']['retained_training'], 6)
            self.assertTrue((output / 'cifar_privacy_summary.csv').is_file())
            with (output / 'cifar_per_class_summary.csv').open() as stream:
                per_class = list(csv.DictReader(stream))
            gold = [row for row in per_class if row['model'] == 'gold']
            self.assertEqual(len(gold), 3)
            self.assertEqual([float(row['forget_accuracy_pct_mean']) for row in gold[:2]],
                             [100.0, 0.0])
            self.assertTrue(all(row['lira_auc_mean'] for row in gold[:2]))
            self.assertEqual(gold[2]['target_count_mean'], '0.0')
            self.assertFalse(gold[2]['lira_auc_mean'])


if __name__ == '__main__':
    unittest.main()
