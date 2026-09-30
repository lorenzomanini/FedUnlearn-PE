"""Guard the CIFAR-10 preferential paper profile and its client semantics."""

import unittest

import numpy as np

from experiments.configs import cifar_preferential_paper
from experiments.datasets import get_clients_subsets


class BalancedLabels:
    targets = [label for label in range(10) for _ in range(100)]

    def __len__(self):
        return len(self.targets)


class CifarPreferentialPaperTests(unittest.TestCase):
    def test_full_lira_budget_and_preferential_partition(self):
        config = cifar_preferential_paper.build_config()
        cases = cifar_preferential_paper.build_cases()
        self.assertEqual(config['dataset_name'], 'cifar10')
        self.assertEqual(config['distribution_type'], 'preferential_class')
        self.assertEqual((config['num_clients'], config['num_classes']), (5, 10))
        self.assertEqual((config['num_tests'], config['num_shadow_models']), (20, 40))
        self.assertEqual(config['repetition_seed'], 3000)
        self.assertEqual(config['train_epochs'], 40)
        self.assertTrue(config['save_models'])
        self.assertEqual(len(cases), 10)
        self.assertEqual((cases[0]['unlearning_percentage'],
                          cases[-1]['unlearning_percentage']), (0.0, 100.0))
        self.assertTrue(all(case['tests'] == ['LiRA'] for case in cases))
        self.assertTrue(all(case['reset_strategy'] == 'zero' and
                            case['retrain_epochs'] == 1 for case in cases))

        np.random.seed(2026)
        labels = np.asarray(BalancedLabels.targets)
        clients = get_clients_subsets(BalancedLabels(), config)
        counts = np.asarray([
            np.bincount(labels[client.indices], minlength=10) for client in clients
        ])
        self.assertEqual(counts.shape, (5, 10))
        self.assertTrue(np.all(counts[:, :5] > 0))
        self.assertTrue(np.all(np.diag(counts[:, 5:]) > 0))
        self.assertEqual(np.count_nonzero(counts[:, 5:]), 5)


if __name__ == '__main__':
    unittest.main()
