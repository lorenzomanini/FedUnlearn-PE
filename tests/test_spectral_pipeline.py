"""Small real training/score/recovery/audit runs with artifact-reader checks."""
import logging
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset

from analysis import results
from experiments import persistence, runner, training


class SpectralPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        runner.set_device(torch.device('cpu'))
        runner.set_batch_sizes(8, 8, 16, 16)
        x = torch.randn(60, 2)
        self.train = TensorDataset(x, (x[:, 0] > x[:, 1]).long())
        x = torch.randn(18, 2)
        self.test = TensorDataset(x, (x[:, 0] > x[:, 1]).long())
        self.clients = [Subset(self.train, list(range(i, i + 20))) for i in (0, 20, 40)]
        self.handlers = list(logging.getLogger().handlers)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        plt.close('all')
        for handler in list(logging.getLogger().handlers):
            if handler not in self.handlers:
                logging.getLogger().removeHandler(handler)
                handler.close()

    def run_suite(self, output, audit, reset_strategy='zero'):
        params = dict(test_name='tiny_pipeline', target_client=0, train_epochs=2,
                      trainer_name='sgd', learning_rate=0.05, momentum=0.0,
                      num_tests=2, num_shadow_models=8 if audit else 0,
                      spectral_max_samples=12, spectral_target_max_samples=8,
                      spectral_rank=4, spectral_num_power_iters=2,
                      spectral_seed=3)
        cases = [dict(unlearning_method='information', unlearning_percentage=p,
                      reset_strategy=reset_strategy,
                      retrain_epochs=1, tests=['LiRA'] if audit and p in (0, 100) else [])
                 for p in (0, 50, 100)]
        trainer = lambda *args: training.revised_simple_trainer(
            *args, init_params_dict=params, train_batch_size=16, eval_batch_size=16)
        run_iteration = lambda i, args: runner._run_tests_iter(
            i, args, runner.SpectralTest, torch.device('cpu'), filter_error_results=True)
        runner._run_repeated_tests(
            params, cases, output, 1, None, False, run_iteration, lambda _: None,
            torch.device('cpu'), lambda _: (self.train, self.test),
            lambda *_: self.clients, lambda _: lambda: nn.Linear(2, 2),
            lambda _: nn.CrossEntropyLoss, lambda _: trainer, lambda *_: None)
        return Path(output) / params['test_name']

    def test_initial_reset_is_recorded_and_used_by_both_recoveries(self):
        with tempfile.TemporaryDirectory() as output, mock.patch.object(
            runner, 'UnlearnNet', wraps=runner.UnlearnNet,
        ) as make_wrapper:
            suite = self.run_suite(output, False, reset_strategy='initial')
            self.assertEqual(make_wrapper.call_count, 8)
            for call in make_wrapper.call_args_list:
                self.assertIsNotNone(call.kwargs['reset_reference'])
            for case in range(0, len(make_wrapper.call_args_list), 2):
                score_reference = make_wrapper.call_args_list[case].kwargs['reset_reference']
                random_reference = make_wrapper.call_args_list[case + 1].kwargs['reset_reference']
                self.assertIs(score_reference, random_reference)
            for repetition in range(2):
                extra = persistence.load_pickle(suite / f'test_{repetition}', persistence.EXTRA_RESULTS)
                self.assertEqual(extra['reset_strategy'], ['initial'] * 3)
                for counts, total in zip(extra['num_reset_per_tensor'], extra['num_reset_params']):
                    self.assertEqual(sum(counts.values()), total)

    def test_complete_pipeline_and_sparse_lira_plots(self):
        for audit in (False, True):
            with self.subTest(audit=audit), tempfile.TemporaryDirectory() as output:
                suite = self.run_suite(output, audit)
                for i in range(2):
                    folder = suite / f'test_{i}'
                    diagnostics = persistence.load_pickle(folder, 'score_diagnostics.pkl')
                    self.assertEqual(diagnostics['full_samples'], 12)
                    self.assertEqual(diagnostics['target_samples'], 8)
                    self.assertEqual(diagnostics['full_population'], 56)
                    self.assertEqual(len(diagnostics['training_indices']), 56)
                    self.assertEqual(diagnostics['target_fraction'], 20 / 56)
                    self.assertTrue(np.isfinite(diagnostics['whitened_target'].numpy()).all())
                    extra = persistence.load_pickle(folder, persistence.EXTRA_RESULTS)
                    self.assertEqual(extra['case_index'], [0, 1, 2])
                    self.assertEqual(extra['num_reset_params'][0], 0)
                    self.assertGreater(extra['num_reset_params'][1], 0)
                    self.assertTrue(all(t >= 0 for t in extra['unlearning_with_score_seconds']))
                    if audit:
                        self.assertEqual(persistence.load_pickle(folder, persistence.LIRA_CASE_INDICES), [0, 2])
                        scores = persistence.load_npz(folder, persistence.EVAL_LIRA_RESULTS)
                        self.assertEqual(scores['retrained'].shape[0], 2)
                        self.assertTrue(np.isfinite(scores['retrained']).all())
                with mock.patch.object(plt, 'show'), mock.patch.object(
                    results, '_plot_metric_vs_unlearning', wraps=results._plot_metric_vs_unlearning
                ) as plot:
                    results.plot_experiment_results(suite, target_fpr=0.1)
                    self.assertEqual(plot.call_count, 2 if audit else 1)
                    self.assertEqual(np.asarray(plot.call_args_list[0].kwargs['x_values']).shape, (3, 2))
                    if audit:
                        self.assertEqual(np.asarray(plot.call_args_list[1].kwargs['x_values']).shape, (2, 2))
                # Never aggregate different successful cases as if they matched.
                folder = suite / 'test_1'
                extra = persistence.load_pickle(folder, persistence.EXTRA_RESULTS)
                extra['case_index'] = [0, 1, 3]
                persistence.dump_pickle(folder, persistence.EXTRA_RESULTS, extra)
                with self.assertRaisesRegex(ValueError, 'cases differ across repetitions'):
                    results.plot_experiment_results(suite)

    def test_fixed_curvature_subsets_and_explicit_full_data(self):
        a = runner._curvature_loader(self.train, 12, 8, 5)
        b = runner._curvature_loader(self.train, 12, 8, 5)
        self.assertEqual(a.dataset.indices, b.dataset.indices)
        self.assertEqual(len(set(a.dataset.indices)), 12)
        self.assertEqual(len(runner._curvature_loader(self.train, 0, 8, 5).dataset), 60)
        self.assertEqual(len(runner._curvature_loader(self.train, None, 8, 5).dataset), 60)
        for value in (-1, True, 2.5):
            with self.assertRaises(ValueError):
                runner._curvature_loader(self.train, value, 8, 5)


if __name__ == '__main__':
    unittest.main()
