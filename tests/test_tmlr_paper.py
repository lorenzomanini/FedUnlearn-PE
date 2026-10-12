"""Guard the TMLR suite registry, its partitions and the Slurm array that runs it."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.configs import tmlr_paper
from experiments.datasets import get_clients_subsets


class BalancedLabels:
    targets = [label for label in range(10) for _ in range(100)]

    def __len__(self):
        return len(self.targets)


def split(**config):
    np.random.seed(2026)
    clients = get_clients_subsets(BalancedLabels(), {'num_clients': 5, 'num_classes': 10, **config})
    return [sorted(client.indices) for client in clients]


class TmlrPaperTests(unittest.TestCase):
    def test_queue_array_covers_every_suite(self):
        text = (Path(__file__).resolve().parents[1] / 'queue.sh').read_text()
        first, last = map(int, re.search(r'#SBATCH --array=(\d+)-(\d+)', text).groups())
        self.assertEqual((first, last), (0, len(tmlr_paper.SUITES) - 1))
        self.assertEqual(tmlr_paper.resolve('0'), next(iter(tmlr_paper.SUITES)))
        with self.assertRaises(KeyError):
            tmlr_paper.resolve(str(len(tmlr_paper.SUITES)))

    def test_suites_share_one_protocol(self):
        for name, suite in tmlr_paper.SUITES.items():
            config = suite['config']
            self.assertEqual(config['test_name'], name)
            self.assertEqual((config['num_clients'], config['target_client']), (5, 0))
            self.assertEqual((config['num_tests'], config['num_shadow_models']), (20, 64))
            self.assertEqual((config['repetition_seed'], config['lira_seed']), (3000, 2026))
            self.assertEqual(config['save_models'], 'baseline')
            if suite['method'] in {'core', 'generalized'}:
                self.assertTrue(config['spectral_control_target'])
            keys = [(case['unlearning_percentage'], case['reset_strategy'],
                     case['recovery_scope'], case['retrain_epochs']) for case in suite['cases']]
            self.assertEqual(len(keys), len(set(keys)))
            # Case 0 is always the unchanged original.
            self.assertEqual(keys[0], (0.0, 'initial', 'selected', 1))
            self.assertTrue(all(case['tests'] == ['LiRA'] for case in suite['cases']))

    def test_main_cases_pair_reset_rules_and_include_finetuning(self):
        keys = [(case['unlearning_percentage'], case['reset_strategy'], case['recovery_scope'])
                for case in tmlr_paper.main_cases()]
        self.assertIn((0.0, 'initial', 'all'), keys)
        for reset in tmlr_paper.RESET_RULES:
            for scope, masses in (('selected', tmlr_paper.MASSES),
                                  ('all', tmlr_paper.FULL_WEIGHT_MASSES)):
                self.assertEqual(sorted(mass for mass, rule, kind in keys
                                        if rule == reset and kind == scope and mass > 0),
                                 sorted(masses))

    def test_core_and_diagonal_suites_are_matched(self):
        for dataset, strength in (('c10', '100'), ('c10', '050'), ('c10', '000'), ('fm', '100')):
            core = tmlr_paper.SUITES[f'{dataset}_mix{strength}_core']
            diagonal = tmlr_paper.SUITES[f'{dataset}_mix{strength}_diag']
            self.assertEqual(core['cases'], diagonal['cases'])
            shared = set(tmlr_paper.base_config('cifar10')) | {'distribution_type', 'preferential_strength'}
            self.assertEqual({key: core['config'][key] for key in shared},
                             {key: diagonal['config'][key] for key in shared})
            self.assertEqual(diagonal['config']['diagonal_population'], 'training')
            self.assertFalse(diagonal['config']['stochastic_correction'])

    def test_generalized_suites_change_only_the_estimator(self):
        estimator = {'spectral_subspace', 'spectral_curvature_backend', 'spectral_num_power_iters',
                     'spectral_damping', 'spectral_rank', 'test_name'}
        for strength in ('100', '050', '000'):
            core = tmlr_paper.SUITES[f'c10_mix{strength}_core']
            generalized = tmlr_paper.SUITES[f'c10_mix{strength}_gen']
            self.assertEqual(core['cases'], generalized['cases'])
            self.assertEqual({key: value for key, value in core['config'].items() if key not in estimator},
                             {key: value for key, value in generalized['config'].items()
                              if key not in estimator and key != 'spectral_damping'})
            self.assertEqual((generalized['config']['spectral_subspace'],
                              generalized['config']['spectral_curvature_backend']), ('generalized', 'ggn'))
        base = tmlr_paper.SUITES['c10_mix000_gen']['config']
        for name, key, value in (('c10_mix000_gen_damp1e-06', 'spectral_damping', 1e-6),
                                 ('c10_mix000_gen_damp1e-02', 'spectral_damping', 1e-2),
                                 ('c10_mix000_gen_rank40', 'spectral_rank', 40)):
            config = tmlr_paper.SUITES[name]['config']
            self.assertEqual(config[key], value)
            self.assertEqual({k: v for k, v in config.items() if k not in {key, 'test_name'}},
                             {k: v for k, v in base.items() if k not in {key, 'test_name'}})
        for strength in ('100', '000'):
            config = tmlr_paper.SUITES[f'c10_mix{strength}_core_ggn']['config']
            self.assertEqual((config['spectral_subspace'], config['spectral_curvature_backend']),
                             ('top_curvature', 'ggn'))

    def test_preferential_mix_interpolates_between_uniform_and_preferential(self):
        labels = np.asarray(BalancedLabels.targets)
        self.assertEqual(split(distribution_type='preferential_mix', preferential_strength=1.0),
                         split(distribution_type='preferential_class'))
        self.assertEqual(split(distribution_type='preferential_mix', preferential_strength=0.0),
                         split(distribution_type='uniform'))
        shares = []
        for strength in (0.0, 0.25, 0.5, 0.75, 1.0):
            clients = split(distribution_type='preferential_mix', preferential_strength=strength)
            sizes = [len(client) for client in clients]
            self.assertEqual(len(set(sizes)), 1)
            # Flooring may drop one record per class and client (0.1% on CIFAR-10).
            self.assertGreaterEqual(sum(sizes), len(labels) - 5 * 10)
            # Share of the exclusive class 5 held by the target client.
            shares.append(np.sum(labels[clients[0]] == 5) / np.sum(labels == 5))
        np.testing.assert_allclose(shares, [0.2, 0.4, 0.6, 0.8, 1.0], atol=0.03)
        for strength in (-0.1, 1.5, True):
            with self.assertRaises(ValueError):
                split(distribution_type='preferential_mix', preferential_strength=strength)

    def test_queue_runs_one_suite_per_array_task(self):
        """Run queue.sh with stubbed cluster commands, without training."""
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            binary = temporary / 'bin'
            binary.mkdir()
            # The suite registry runs for real; other Python calls are environment probes.
            python = binary / 'python'
            python.write_text('#!/bin/sh\n'
                              'if [ "$1" = "-m" ] && [ "$2" = "experiments.configs.tmlr_paper" ]; then\n'
                              '    exec "$REAL_PYTHON" "$@"\nfi\n'
                              "cat >/dev/null\nprintf '%s\\n' '/tmp'\n")
            python.chmod(0o755)
            (binary / 'activate').write_text('')
            stub = temporary / 'cluster.sh'
            stub.write_text('module() { :; }\nsrun() { printf "%s\\0" "$@" >> "$QUEUE_CALLS"; printf "\\n" >> "$QUEUE_CALLS"; }\n')
            last = len(tmlr_paper.SUITES) - 1
            for task, suite in ((None, None), ('0', 'c10_mix100_core'),
                                (str(last), list(tmlr_paper.SUITES)[last]),
                                (str(last + 1), None)):
                with self.subTest(task=task):
                    calls = temporary / f'calls_{task}'
                    environment = dict(os.environ, PROJECT_DIR=str(root), VENV_PATH=str(temporary),
                                       OUTPUT_ROOT=str(temporary / 'results'), REAL_PYTHON=sys.executable,
                                       SLURM_ARRAY_JOB_ID='123', SLURM_CPUS_PER_TASK='1',
                                       PATH=f'{binary}{os.pathsep}{os.environ["PATH"]}',
                                       QUEUE_CALLS=str(calls), QUEUE_STUB=str(stub),
                                       QUEUE_SCRIPT=str(root / 'queue.sh'))
                    environment.pop('SLURM_ARRAY_TASK_ID', None)
                    if task is not None:
                        environment['SLURM_ARRAY_TASK_ID'] = task
                    result = subprocess.run(
                        ['bash', '-c', 'source "$QUEUE_STUB"; source "$QUEUE_SCRIPT"'],
                        env=environment, capture_output=True, text=True,
                    )
                    if suite is None:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertFalse(calls.exists())
                        continue
                    self.assertEqual(result.returncode, 0, result.stderr)
                    output = temporary / 'results' / f'{suite}_job_123'
                    self.assertEqual(json.loads((output / 'suite.json').read_text())['suite'], suite)
                    smoke, experiment, audit = [line.split('\0')[:-1]
                                                for line in calls.read_text().splitlines()]
                    self.assertEqual(smoke[smoke.index('--suite') + 1], suite)
                    self.assertIn('--smoke', smoke)
                    self.assertEqual(experiment[-4:], ['--suite', suite, '--output', str(output)])
                    self.assertIn('--allow-inefficient', audit)
                    self.assertEqual(audit[audit.index('--input') + 1], str(output / suite))

    def test_run_refuses_an_existing_suite_directory(self):
        name = next(iter(tmlr_paper.SUITES))
        with tempfile.TemporaryDirectory() as output, \
                mock.patch.object(tmlr_paper.spectral_runner, 'run_repeated_tests') as run:
            (Path(output) / name).mkdir()
            with self.assertRaises(SystemExit):
                tmlr_paper.run(name, Path(output))
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
