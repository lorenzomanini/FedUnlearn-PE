"""Exercise the diagnostic protocol on real tiny Hessians and recovery updates."""

import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments import cifar_diagnosis as diagnosis
from experiments.cifar_recovery_ablation import recover
from fisherunlearn.information.selection import find_informative_params_matching


class LabeledDataset(TensorDataset):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.targets = y


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)

    def test_pseudo_ranking_matches_layer_count_across_weight_and_bias(self):
        scores = {'0.weight': torch.tensor([[1., 2.], [3., 4.]]),
                  '0.bias': torch.tensor([9., 8.]), '1.weight': torch.tensor([1., 5.])}
        reference = {'0.weight': torch.tensor([[0, 0], [1, 1]]),
                     '0.bias': torch.empty(0, 1, dtype=torch.long), '1.weight': torch.tensor([[0]])}
        result = find_informative_params_matching(scores, reference)
        self.assertEqual(len(result['0.weight']), 0)
        torch.testing.assert_close(result['0.bias'], torch.tensor([[0], [1]]))
        torch.testing.assert_close(result['1.weight'], torch.tensor([[1]]))

    def test_dense_frozen_bn_callback_and_momentum_match_uninterrupted_training(self):
        model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.Tanh(), nn.Linear(3, 2))
        control = copy.deepcopy(model)
        x = torch.randn(16, 2)
        dataset = TensorDataset(x, (x[:, 0] > 0).long())
        loader = DataLoader(dataset, batch_size=8)
        snapshots = []
        before = copy.deepcopy(model.state_dict())
        recover(model, loader, loader, epochs=2, learning_rate=.02, momentum=.9,
                device=torch.device('cpu'), deadline=float('inf'), batchnorm_mode='frozen',
                on_epoch=lambda epoch, net, history: snapshots.append(epoch))
        optimizer = torch.optim.SGD(control.parameters(), lr=.02, momentum=.9)
        for _ in range(2):
            control.train()
            control[1].eval()
            for inputs, targets in loader:
                optimizer.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(control(inputs), targets).backward()
                optimizer.step()
        self.assertEqual(snapshots, [1, 2])
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, control.state_dict()[key], rtol=0, atol=0)
        for key in ('1.running_mean', '1.running_var', '1.num_batches_tracked'):
            torch.testing.assert_close(model.state_dict()[key], before[key], rtol=0, atol=0)

    def test_tiny_checkpoint_protocol_preserves_controls_and_audits_actual_target(self):
        x = torch.randn(40, 2)
        train = LabeledDataset(x, (x[:, 0] > 0).long())
        test = LabeledDataset(x[:8], (x[:8, 0] > 0).long())
        make_model = lambda: nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.Tanh(), nn.Linear(4, 2))
        model = make_model().eval()
        initial = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            model[0].weight.add_(.1)
            model[1].bias.add_(.1)
        before = copy.deepcopy(model.state_dict())
        device = torch.device('cpu')
        train_result = diagnosis.evaluate_model(model, train, device, 8)
        test_result = diagnosis.evaluate_model(model, test, device, 8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bootstrap = root / 'bootstrap'
            folder = bootstrap / 'suite' / 'test_0'
            folder.mkdir(parents=True)
            artifacts = [
                (folder.parent / 'init_params.pkl', dict(dataset_name='cifar10', model_name='resnet18',
                                                        learning_rate=.01, momentum=.9, lira_global_variance=False)),
                (folder.parent / 'labels.pkl', {'train': train.targets.numpy(), 'test': test.targets.numpy()}),
                (folder / 'score_diagnostics.pkl', {'training_indices': list(range(32)), 'target_indices': list(range(8))}),
                (folder / 'initial_eval_train_results.pkl', {'trained': train_result, 'shadow_out': train_result}),
                (folder / 'initial_eval_test_results.pkl', {'trained': test_result, 'shadow_out': test_result}),
            ]
            for path, value in artifacts:
                with path.open('wb') as stream:
                    pickle.dump(value, stream)
            shadow_membership = np.tile(np.array([True, True, False, False])[:, None], (1, 16))
            np.savez(folder.parent / 'lira_shadow_bank.npz',
                     candidate_complete_indices=np.array(list(range(8)) + list(range(40, 48))),
                     candidate_membership=np.array([True] * 8 + [False] * 8),
                     shadow_membership=shadow_membership,
                     scores=np.random.default_rng(9).normal(size=(4, 16)) + shadow_membership * .5)
            torch.save(before, bootstrap / 'original_model.pth')
            torch.save(initial, bootstrap / 'initial_model.pth')
            output = root / 'controls'
            arguments = ['diagnosis', '--suite', str(folder.parent), '--checkpoint', str(bootstrap / 'original_model.pth'),
                         '--output', str(output), '--rank', '2', '--higher-rank', '4', '--baseline-power-iters', '1',
                         '--power-iters', '2', '--max-samples', '8', '--epochs', '2', '--batch-size', '8']
            with mock.patch('sys.argv', arguments), mock.patch.object(
                diagnosis, 'get_datasets', return_value=(train, test),
            ), mock.patch.object(diagnosis, 'get_model_class', return_value=make_model):
                diagnosis.main()
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['status'], 'completed_diagnostic_controls_pending_analysis')
            self.assertEqual(len(report['rows']), 2 + 6 * 3)
            split = torch.load(output / 'split.pt', weights_only=False)
            self.assertEqual(len(split['pseudo_target_indices']), 8)
            self.assertFalse(set(split['pseudo_target_indices']) & set(range(8)))
            self.assertEqual(split['validation_indices'], list(range(32, 40)))
            self.assertEqual(report['scores']['pseudo']['same_basis_as'], 'reference')
            masks = {name: torch.load(output / f'{name}_mask.pt', weights_only=True)
                     for name in ('actual_selected', 'pseudo_selected', 'random_selected')}
            for layer in ('0', '1', '3'):
                counts = [sum(len(value) for key, value in mask.items() if key.startswith(layer + '.'))
                          for mask in masks.values()]
                self.assertEqual(len(set(counts)), 1)
            resets = [torch.load(output / name / 'epoch_0' / 'model.pth', weights_only=True)
                      for name in ('actual_selected', 'actual_full', 'actual_frozen_bn')]
            for key in before:
                for state in resets[1:]:
                    torch.testing.assert_close(state[key], resets[0][key], rtol=0, atol=0)
            selected = torch.load(output / 'actual_selected' / 'epoch_2' / 'model.pth', weights_only=True)
            frozen = torch.load(output / 'actual_frozen_bn' / 'epoch_2' / 'model.pth', weights_only=True)
            for key, _ in model.named_parameters():
                unchanged = torch.ones_like(before[key], dtype=torch.bool)
                unchanged[tuple(masks['actual_selected'][key].t())] = False
                torch.testing.assert_close(selected[key][unchanged], before[key][unchanged], rtol=0, atol=0)
            for key in ('1.running_mean', '1.running_var', '1.num_batches_tracked'):
                torch.testing.assert_close(frozen[key], resets[0][key], rtol=0, atol=0)
            self.assertEqual(selected['1.num_batches_tracked'].item(), 6)
            self.assertEqual(set(report['mask_comparisons']),
                             {'baseline_vs_reference', 'reference_vs_higher_rank', 'actual_vs_pseudo', 'actual_vs_random'})
            with np.load(output / 'pseudo_selected' / 'epoch_2' / 'privacy_scores.npz') as audit:
                self.assertEqual(audit['confidence'].shape, (16,))
                self.assertTrue(np.isfinite(audit['llr']).all())
            for key, value in torch.load(bootstrap / 'original_model.pth', weights_only=True).items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)

            # Reuse exact artifacts for the follow-up; a curvature call is a bug.
            source_hashes = {path: diagnosis.file_hash(path) for path in
                             [output / 'report.json', bootstrap / 'original_model.pth', bootstrap / 'initial_model.pth']}
            followup = root / 'followup' / 'controls'
            arguments = ['diagnosis', '--reuse-diagnosis', str(root), '--output', str(followup)]
            with mock.patch('sys.argv', arguments), mock.patch.object(
                diagnosis, 'get_datasets', return_value=(train, test),
            ), mock.patch.object(diagnosis, 'get_model_class', return_value=make_model), mock.patch.object(
                diagnosis, 'estimate_core_score', side_effect=AssertionError('Follow-up must not recompute scores'),
            ), mock.patch.object(diagnosis, 'make_hessian_matvec', side_effect=AssertionError('Follow-up must not compute HVPs')):
                diagnosis.main()
            follow_report = json.loads((followup / 'report.json').read_text())
            self.assertEqual(follow_report['mode'], 'recovery_controls')
            self.assertEqual(follow_report['total_scoring_seconds'], 0)
            self.assertEqual(set(follow_report['conditions']), {'no_reset_full', 'actual_full', 'pseudo_full', 'random_full'})
            self.assertEqual(len(follow_report['rows']), 2 + 4 * 3)
            self.assertEqual(follow_report['retained_batch_seed'], report['retained_batch_seed'])
            self.assertEqual(follow_report['arguments']['batch_size'], 8)
            self.assertEqual(follow_report['provenance'], report['provenance'])
            no_reset_condition = follow_report['conditions']['no_reset_full']
            self.assertEqual(no_reset_condition['trainable'], 'all')
            self.assertEqual(sum(no_reset_condition['selected_per_tensor'].values()), 0)
            self.assertEqual([row['epoch'] for row in no_reset_condition['history']], [1, 2])
            no_reset = torch.load(followup / 'no_reset_full/epoch_0/model.pth', weights_only=True)
            for key in before:
                torch.testing.assert_close(no_reset[key], before[key], rtol=0, atol=0)
            no_reset_recovered = torch.load(followup / 'no_reset_full/epoch_2/model.pth', weights_only=True)
            self.assertTrue(any(not torch.equal(no_reset_recovered[key], before[key]) for key, _ in model.named_parameters()))
            for new, old in [('actual_full', 'actual_full'), ('pseudo_full', 'pseudo_selected'), ('random_full', 'random_selected')]:
                reset_state = torch.load(followup / new / 'epoch_0/model.pth', weights_only=True)
                source_state = torch.load(output / old / 'epoch_0/model.pth', weights_only=True)
                for key in source_state:
                    torch.testing.assert_close(reset_state[key], source_state[key], rtol=0, atol=0)
            # Same reset, optimizer and batch sequence reproduce the CPU comparator.
            for epoch in (1, 2):
                current = torch.load(followup / 'actual_full' / f'epoch_{epoch}/model.pth', weights_only=True)
                previous = torch.load(output / 'actual_full' / f'epoch_{epoch}/model.pth', weights_only=True)
                for key in previous:
                    torch.testing.assert_close(current[key], previous[key], rtol=0, atol=0)
            for path, digest in source_hashes.items():
                self.assertEqual(diagnosis.file_hash(path), digest)
            # Refuse a mismatch before creating another result directory.
            corrupted = copy.deepcopy(report)
            corrupted['provenance']['checkpoint_sha256'] = 'wrong-checkpoint'
            (output / 'report.json').write_text(json.dumps(corrupted))
            bad_output = root / 'mismatched_output'
            with mock.patch('sys.argv', ['diagnosis', '--reuse-diagnosis', str(root), '--output', str(bad_output)]), mock.patch.object(
                diagnosis, 'get_datasets', return_value=(train, test),
            ), mock.patch.object(diagnosis, 'get_model_class', return_value=make_model), self.assertRaisesRegex(ValueError, 'differs from the source'):
                diagnosis.main()
            self.assertFalse(bad_output.exists())


if __name__ == '__main__':
    unittest.main()
