"""Small real training/score/recovery/audit runs with artifact-reader checks."""
import csv
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
from validation.accuracy_audit import analyze_saved_cifar


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

    def run_suite(self, output, audit, reset_strategy='zero', initial_states=None,
                  cases=None, save_models=False, test_class=runner.SpectralTest,
                  trained_types=None, **extra_params):
        params = dict(test_name='tiny_pipeline', target_client=0, train_epochs=2,
                      trainer_name='sgd', learning_rate=0.05, momentum=0.0,
                      num_tests=2, num_shadow_models=8 if audit else 0,
                      spectral_max_samples=12, spectral_target_max_samples=8,
                      spectral_rank=4, spectral_num_power_iters=2,
                      spectral_seed=3)
        params.update(extra_params)
        if cases is None:
            cases = [dict(unlearning_method='information', unlearning_percentage=p,
                          reset_strategy=reset_strategy,
                          retrain_epochs=1, tests=['LiRA'] if audit and p in (0, 100) else [])
                     for p in (0, 50, 100)]
        def trainer(*args):
            if initial_states is not None and args[4] == params['train_epochs']:
                initial_states.append({name: value.detach().clone()
                                       for name, value in args[0].state_dict().items()})
            if trained_types is not None:
                trained_types.append(type(args[0]).__name__)
            return training.revised_simple_trainer(
                *args, init_params_dict=params, train_batch_size=16, eval_batch_size=16)
        run_iteration = lambda i, args: runner._run_tests_iter(
            i, args, test_class, torch.device('cpu'), filter_error_results=True)
        runner._run_repeated_tests(
            params, cases, output, 1, None, save_models, run_iteration, lambda _: None,
            torch.device('cpu'), lambda _: (self.train, self.test),
            lambda *_: self.clients, lambda _: lambda: nn.Linear(2, 2),
            lambda _: nn.CrossEntropyLoss, lambda _: trainer, lambda *_: None)
        return Path(output) / params['test_name']

    def test_initial_reset_is_recorded_and_used_by_both_recoveries(self):
        initial_states = []
        with tempfile.TemporaryDirectory() as output, mock.patch.object(
            runner, 'UnlearnNet', wraps=runner.UnlearnNet,
        ) as make_wrapper:
            suite = self.run_suite(output, False, reset_strategy='initial', initial_states=initial_states)
            self.assertEqual(make_wrapper.call_count, 8)
            for call in make_wrapper.call_args_list:
                self.assertIsNotNone(call.kwargs['reset_reference'])
            for case in range(0, len(make_wrapper.call_args_list), 2):
                score_reference = make_wrapper.call_args_list[case].kwargs['reset_reference']
                random_reference = make_wrapper.call_args_list[case + 1].kwargs['reset_reference']
                self.assertIs(score_reference, random_reference)
                original_state = initial_states[2 * (case // 4)]
                for name, value in original_state.items():
                    torch.testing.assert_close(score_reference[name], value, rtol=0, atol=0)
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

    def test_full_weight_recovery_finetuning_baseline_control_and_checkpoints(self):
        cases = [dict(unlearning_method='information', unlearning_percentage=mass,
                      reset_strategy=reset, recovery_scope=scope, retrain_epochs=1,
                      tests=['LiRA'])
                 for mass, reset, scope in ((0, 'initial', 'selected'), (50, 'initial', 'all'),
                                            (50, 'zero', 'selected'), (0, 'initial', 'all'))]
        trained_types = []
        with tempfile.TemporaryDirectory() as output:
            suite = self.run_suite(output, True, cases=cases, save_models='baseline',
                                   trained_types=trained_types, spectral_control_target=True)
            # 8 shadows; per repetition original and gold, two full-weight recoveries,
            # two selected-only (TRIM) recoveries and one no-reset fine-tune.
            self.assertEqual(len(trained_types), 8 + 2 * 7)
            self.assertEqual(trained_types.count('UnlearnNet'), 2 * 2)
            for repetition in range(2):
                folder = suite / f'test_{repetition}'
                extra = persistence.load_pickle(folder, persistence.EXTRA_RESULTS)
                self.assertEqual(extra['recovery_scope'], ['selected', 'all', 'selected', 'all'])
                self.assertEqual(extra['num_reset_params'][3], 0)
                self.assertGreater(extra['recovery_seconds'][3], 0)
                train = persistence.load_npz(folder, persistence.EVAL_TRAIN_RESULTS)
                original = persistence.load_pickle(folder, persistence.INITIAL_TRAIN_RESULTS)['trained']
                np.testing.assert_array_equal(train['reset__loss'][3], original['loss'])
                np.testing.assert_array_equal(train['retrained__loss'][3], train['random_retrained__loss'][3])
                self.assertFalse(np.array_equal(train['retrained__loss'][3], original['loss']))
                scores = persistence.load_npz(folder, persistence.EVAL_LIRA_RESULTS)
                np.testing.assert_array_equal(scores['retrained'][3], scores['random_retrained'][3])
                diagnostics = persistence.load_pickle(folder, 'score_diagnostics.pkl')
                control = diagnostics['diagnostics']['control']
                self.assertEqual(diagnostics['control_samples'], 8)
                self.assertTrue(0 <= control['top_set_overlap'] <= 1)
                self.assertEqual(sorted(path.name for path in folder.glob('*.pth')),
                                 ['gold_model.pth', 'initial_model.pth', 'original_model.pth'])
            with mock.patch('sys.argv', ['audit', '--input', str(suite),
                                         '--output', str(Path(output) / 'summary'),
                                         '--allow-inefficient']):
                analyze_saved_cifar.main()
            with (Path(output) / 'summary' / 'cifar_privacy_summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({(row['case_index'], row['recovery_scope']) for row in rows
                              if row['model'] == 'retrained'},
                             {('0', 'selected'), ('1', 'all'), ('2', 'selected'), ('3', 'all')})

    def test_generalized_subspace_respects_the_bound_and_keeps_the_control_outside_h(self):
        cases = [dict(unlearning_method='information', unlearning_percentage=mass,
                      retrain_epochs=1, tests=['LiRA']) for mass in (0, 50)]
        with tempfile.TemporaryDirectory() as output:
            suite = self.run_suite(output, True, cases=cases, spectral_subspace='generalized',
                                   spectral_control_target=True, spectral_num_power_iters=5)
            for repetition in range(2):
                diagnostics = persistence.load_pickle(suite / f'test_{repetition}', 'score_diagnostics.pkl')
                inner = diagnostics['diagnostics']
                self.assertEqual((diagnostics['spectral_subspace'], inner['subspace_mode'],
                                  inner['curvature_backend']), ('generalized', 'generalized', 'ggn'))
                # H uses 12 retained records plus the 8 target-sample records.
                self.assertEqual((diagnostics['retained_samples'], diagnostics['target_samples'],
                                  diagnostics['full_samples'], diagnostics['control_samples']),
                                 (12, 8, 20, 8))
                self.assertFalse(set(diagnostics['retained_sample_positions'])
                                 & set(diagnostics['control_sample_positions']))
                bound = inner['relative_curvature_bound']
                self.assertAlmostEqual(bound, 56 / 20)
                self.assertLessEqual(torch.linalg.eigvalsh(diagnostics['whitened_target']).max().item(),
                                     bound + 1e-5)
                extra = persistence.load_pickle(suite / f'test_{repetition}', persistence.EXTRA_RESULTS)
                self.assertGreater(extra['num_reset_params'][1], 0)
            with mock.patch('sys.argv', ['audit', '--input', str(suite),
                                         '--output', str(Path(output) / 'summary'),
                                         '--allow-inefficient']):
                analyze_saved_cifar.main()
            with (Path(output) / 'summary' / 'cifar_score_diagnostics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row['spectral_subspace'] for row in rows}, {'generalized'})
            self.assertTrue(all(float(row['max_whitened_target_eigenvalue'])
                                <= float(row['relative_curvature_bound']) + 1e-5 for row in rows))

    def test_diagonal_population_selects_scored_records(self):
        calls = []

        def information(client_index, model, criterion, datasets_list, **kwargs):
            calls.append((client_index, [len(dataset) for dataset in datasets_list]))
            return {name: torch.zeros_like(value) for name, value in model.named_parameters()}

        cases = [dict(unlearning_method='information', unlearning_percentage=50,
                      retrain_epochs=1, tests=[])]
        # Three 20-record clients; retained clients train on 18 and validate on 2.
        for population, expected in (('clients', (0, [20, 20, 20])),
                                     ('training', (2, [18, 18, 20]))):
            calls.clear()
            with self.subTest(population=population), tempfile.TemporaryDirectory() as output, \
                    mock.patch.object(runner, 'compute_client_information', side_effect=information):
                suite = self.run_suite(output, False, cases=cases, test_class=runner.Test,
                                       hessian_method='diag_ggn_mc',
                                       diagonal_population=population)
                self.assertEqual(calls, [expected, expected])
                diagnostics = persistence.load_pickle(suite / 'test_0', 'score_diagnostics.pkl')
                self.assertEqual(diagnostics['diagonal_population'], population)
                self.assertEqual(diagnostics['full_samples'], sum(expected[1]))

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
