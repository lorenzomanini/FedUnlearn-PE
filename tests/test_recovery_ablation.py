"""The cheap recovery path must stop on budget and use retained data only."""

import copy
import json
import pickle
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.cifar_recovery_ablation import recover
from experiments import cifar_recovery_ablation as ablation
from experiments import cifar_bootstrap
from fisherunlearn.unlearning import UnlearnNet


class RecoveryAblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.base = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.ReLU(), nn.Linear(3, 2))
        self.selected = {'0.weight': torch.tensor([[0, 0], [1, 1]])}
        x = torch.randn(16, 2)
        self.loader = DataLoader(TensorDataset(x, (x[:, 0] > 0).long()), batch_size=8)

    def call(self, model, deadline, batchnorm_mode='train'):
        return recover(model, self.loader, self.loader, epochs=2, learning_rate=.01,
                       momentum=.9, device=torch.device('cpu'), deadline=deadline,
                       batchnorm_mode=batchnorm_mode)

    def test_expired_budget_does_not_change_state(self):
        model = UnlearnNet(self.base, self.selected)
        before = copy.deepcopy(model.state_dict())
        with self.assertRaises(TimeoutError):
            self.call(model, time.perf_counter() - 1)
        for name, value in before.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    def test_recovery_changes_only_selected_weights_and_handles_bn_ablation(self):
        for mode in ('train', 'frozen'):
            model = UnlearnNet(self.base, self.selected, copy.deepcopy(self.base))
            before = model.get_retrained_params()
            history = self.call(model, time.perf_counter() + 30, mode)
            self.assertEqual([row['epoch'] for row in history], [1, 2])
            after = model.get_retrained_params()
            for name, _ in self.base.named_parameters():
                mask = torch.ones_like(before[name], dtype=torch.bool)
                if name in self.selected:
                    mask[tuple(self.selected[name].t())] = False
                torch.testing.assert_close(before[name][mask], after[name][mask], rtol=0, atol=0)
            expected_batches = 4 if mode == 'train' else 0
            self.assertEqual(after['1.num_batches_tracked'].item(), expected_batches)

    def test_checkpoint_command_completes_without_training_baselines(self):
        x = torch.randn(40, 2)
        dataset = TensorDataset(x, (x[:, 0] > 0).long())
        make_model = lambda: nn.Linear(2, 2)
        model = make_model()
        outputs = ablation.evaluate_model(model, dataset, torch.device('cpu'), 8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'test_0'
            folder.mkdir()
            config = dict(dataset_name='cifar10', model_name='resnet18',
                          learning_rate=.01, momentum=.9)
            for path, value in [
                (root / 'init_params.pkl', config),
                (root / 'labels.pkl', {'train': dataset.tensors[1].numpy(), 'test': dataset.tensors[1].numpy()}),
                (folder / 'score_diagnostics.pkl', {'training_indices': list(range(32)), 'target_indices': list(range(8))}),
                (folder / 'stage_timings.pkl', {'gold_retraining_seconds': 300.0}),
                (folder / 'initial_eval_train_results.pkl', {'trained': outputs, 'shadow_out': outputs}),
                (folder / 'initial_eval_test_results.pkl', {'trained': outputs, 'shadow_out': outputs}),
            ]:
                with path.open('wb') as stream:
                    pickle.dump(value, stream)
            checkpoint = root / 'original.pth'
            torch.save(model.state_dict(), checkpoint)
            output = root / 'ablation'
            arguments = ['ablation', '--suite', str(root), '--checkpoint', str(checkpoint),
                         '--output', str(output), '--epochs', '1', '--rank', '2',
                         '--max-samples', '8', '--target-max-samples', '8', '--batch-size', '8']
            with mock.patch('sys.argv', arguments), mock.patch.object(
                ablation, 'get_datasets', return_value=(dataset, dataset),
            ), mock.patch.object(ablation, 'get_model_class', return_value=make_model):
                ablation.main()
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['status'], 'completed_pending_utility_and_privacy_review')
            self.assertEqual(len(report['history']), 1)
            self.assertEqual(set(report['utility']), {'test', 'retained', 'forget'})
            self.assertLess(report['cost_fraction_of_gold'], .25)
            self.assertTrue((output / 'recovered_model.pth').exists())
            # The supplied original checkpoint is unchanged.
            for key, value in torch.load(checkpoint, weights_only=True).items():
                torch.testing.assert_close(value, model.state_dict()[key])

    def test_bootstrap_saves_fresh_original_and_reuses_gold(self):
        class LabeledDataset(TensorDataset):
            def __init__(self, x, y):
                super().__init__(x, y)
                self.targets = y

        x = torch.randn(24, 2)
        train = LabeledDataset(x, (x[:, 0] > 0).long())
        test = LabeledDataset(x[:6], (x[:6, 0] > 0).long())
        make_model = lambda: nn.Linear(2, 2)
        gold = make_model()
        gold_train = ablation.evaluate_model(gold, train, torch.device('cpu'), 8)
        gold_test = ablation.evaluate_model(gold, test, torch.device('cpu'), 8)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source'
            source_iteration = source / 'test_0'
            source_iteration.mkdir(parents=True)
            artifacts = [
                (source / 'init_params.pkl', dict(dataset_name='cifar10', model_name='resnet18',
                                                  learning_rate=.01, momentum=.9, train_epochs=1)),
                (source / 'labels.pkl', {'train': train.targets.numpy(), 'test': test.targets.numpy()}),
                (source_iteration / 'score_diagnostics.pkl', {
                    'training_indices': list(range(20)), 'target_indices': list(range(4))}),
                (source_iteration / 'stage_timings.pkl', {'gold_retraining_seconds': 3.0}),
                (source_iteration / 'initial_eval_train_results.pkl', {
                    'trained': gold_train, 'shadow_out': gold_train}),
                (source_iteration / 'initial_eval_test_results.pkl', {
                    'trained': gold_test, 'shadow_out': gold_test}),
            ]
            for path, value in artifacts:
                with path.open('wb') as stream:
                    pickle.dump(value, stream)
            output = Path(temporary) / 'bootstrap'
            with mock.patch.object(cifar_bootstrap, 'get_datasets', return_value=(train, test)), mock.patch.object(
                cifar_bootstrap, 'get_model_class', return_value=make_model,
            ):
                checkpoint, suite = cifar_bootstrap.bootstrap(source, 0, output, batch_size=8)
            self.assertTrue(checkpoint.exists())
            self.assertTrue((output / 'manifest.json').exists())
            with (suite / 'test_0' / 'stage_timings.pkl').open('rb') as stream:
                self.assertEqual(pickle.load(stream)['gold_retraining_seconds'], 3.0)
            with (suite / 'test_0' / 'initial_eval_train_results.pkl').open('rb') as stream:
                train_results = pickle.load(stream)
            self.assertEqual(len(train_results['trained']['pred']), len(train))
            self.assertTrue((train_results['shadow_out']['pred'] == gold_train['pred']).all())
            with (suite / 'test_0' / 'score_diagnostics.pkl').open('rb') as stream:
                reduced = pickle.load(stream)
            self.assertEqual(set(reduced), {'training_indices', 'target_indices'})

    def test_fresh_bootstrap_needs_no_saved_suite_and_feeds_ablation(self):
        class LabeledDataset(TensorDataset):
            def __init__(self, x, y):
                super().__init__(x, y)
                self.targets = y

        x = torch.randn(40, 2)
        train = LabeledDataset(x, (x[:, 0] > 0).long())
        test = LabeledDataset(x[:8], (x[:8, 0] > 0).long())
        make_model = lambda: nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'bootstrap'
            with mock.patch.object(cifar_bootstrap, 'get_datasets', return_value=(train, test)), mock.patch.object(
                cifar_bootstrap, 'get_model_class', return_value=make_model,
            ):
                checkpoint, suite = cifar_bootstrap.bootstrap_fresh(
                    output, seed=7, batch_size=8, train_epochs=1,
                )
            self.assertTrue(checkpoint.exists())
            self.assertTrue((output / 'gold_model.pth').exists())
            with (suite / 'test_0' / 'score_diagnostics.pkl').open('rb') as stream:
                split = pickle.load(stream)
            self.assertEqual(len(split['target_indices']), 4)
            self.assertEqual(len(split['training_indices']), 37)
            self.assertTrue(set(split['target_indices']).issubset(split['training_indices']))
            with (suite / 'test_0' / 'stage_timings.pkl').open('rb') as stream:
                timings = pickle.load(stream)
            self.assertGreater(timings['gold_retraining_seconds'], 0)
            self.assertFalse((suite / 'lira_shadow_bank.npz').exists())

            # The deletion runner must accept the newly trained checkpoint and
            # its saved predictions, without importing any prior CIFAR results.
            timings['gold_retraining_seconds'] = 300.0
            with (suite / 'test_0' / 'stage_timings.pkl').open('wb') as stream:
                pickle.dump(timings, stream)
            result = Path(temporary) / 'ablation'
            arguments = ['ablation', '--suite', str(suite), '--checkpoint', str(checkpoint),
                         '--output', str(result), '--epochs', '1', '--rank', '2',
                         '--max-samples', '8', '--target-max-samples', '4', '--batch-size', '8']
            with mock.patch('sys.argv', arguments), mock.patch.object(
                ablation, 'get_datasets', return_value=(train, test),
            ), mock.patch.object(ablation, 'get_model_class', return_value=make_model):
                ablation.main()
            report = json.loads((result / 'report.json').read_text())
            self.assertEqual(report['status'], 'completed_pending_utility_and_privacy_review')
            self.assertEqual(set(report['utility']), {'test', 'retained', 'forget'})


if __name__ == '__main__':
    unittest.main()
