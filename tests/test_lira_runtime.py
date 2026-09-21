"""Bounded LiRA training, sparse audit artifacts, and membership regressions."""

import logging
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset

from experiments import persistence, runner
from experiments.configs import revised_diagonal, spectral_wip


class LiRARuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.dataset = TensorDataset(torch.randn(36, 2), torch.arange(36) % 2)
        self.test_dataset = TensorDataset(torch.randn(12, 2), torch.arange(12) % 2)
        self.clients = [Subset(self.dataset, list(range(start, start + 12)))
                        for start in (0, 12, 24)]
        self.params = {
            "test_name": "tiny_lira", "num_tests": 3, "num_clients": 3,
            "target_client": 0, "trainer_name": "sgd", "train_epochs": 1,
            "learning_rate": 0.05, "momentum": 0.0,
        }
        self.handlers = list(logging.getLogger().handlers)
        self.addCleanup(self._close_new_log_handlers)

    def _close_new_log_handlers(self):
        for handler in list(logging.getLogger().handlers):
            if handler not in self.handlers:
                logging.getLogger().removeHandler(handler)
                handler.close()

    def _run_suite(self, directory, cases, trainer, iteration, params=None):
        runner._run_repeated_tests(
            self.params if params is None else params, cases, directory, 1, None,
            False, iteration, lambda _: None, torch.device("cpu"),
            lambda _: (self.dataset, self.test_dataset),
            lambda *_: self.clients, lambda _: lambda: nn.Linear(2, 2),
            lambda _: nn.CrossEntropyLoss, lambda _: trainer, lambda *_: None,
            evaluate_lira_function=lambda model, dataset: runner._evaluation.evaluate_lira(
                model, dataset, "cpu", 8
            ),
        )

    def test_no_shadow_training_without_explicit_lira(self):
        trainer = mock.Mock(side_effect=AssertionError("unexpected shadow training"))
        iteration = mock.Mock(return_value=[])
        with tempfile.TemporaryDirectory() as directory:
            self._run_suite(directory, [{"tests": ["test_accuracy"]}, {}], trainer, iteration)
            suite = Path(directory) / self.params["test_name"]
            self.assertFalse((suite / persistence.LIRA_SHADOW_BANK).exists())
            saved = persistence.load_pickle(suite, persistence.INIT_PARAMS)
            self.assertFalse(saved["lira_enabled"])
            self.assertEqual(saved["num_shadow_models"], 8)
        self.assertEqual(iteration.call_count, 3)
        trainer.assert_not_called()

    def test_zero_shadows_disables_requested_audit(self):
        trainer = mock.Mock(side_effect=AssertionError("unexpected shadow training"))
        iteration = mock.Mock(return_value=[])
        with tempfile.TemporaryDirectory() as directory:
            self._run_suite(
                directory, [{"tests": ["LiRA"]}], trainer, iteration,
                dict(self.params, num_shadow_models=0),
            )
            self.assertIsNone(iteration.call_args.args[1]["lira_context"])
        trainer.assert_not_called()

    def test_eight_shadows_shared_across_repeats_with_valid_memberships(self):
        trained_indices = []

        def train(model, loss_fn, train_subsets, val_subsets, epochs):
            self.assertEqual(epochs, self.params["train_epochs"])
            self.assertEqual(val_subsets, [])
            subset = train_subsets[0]
            trained_indices.append(set(subset.indices))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
            x, y = next(iter(torch.utils.data.DataLoader(subset, batch_size=len(subset))))
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()
            return model.eval()

        iteration = mock.Mock(return_value=[])
        with tempfile.TemporaryDirectory() as directory:
            self._run_suite(directory, [{"tests": ["LiRA"]}, {}], train, iteration)
            bank = persistence.load_npz(
                Path(directory) / self.params["test_name"], persistence.LIRA_SHADOW_BANK
            )
        self.assertEqual(len(trained_indices), 8)
        self.assertEqual(iteration.call_count, 3)
        self.assertTrue(np.isfinite(bank["scores"]).all())
        self.assertEqual(bank["scores"].shape, bank["shadow_membership"].shape)
        np.testing.assert_array_equal(bank["shadow_membership"].sum(axis=0), 4)
        np.testing.assert_array_equal(
            bank["shadow_membership"].sum(axis=1), len(bank["candidate_membership"]) // 2
        )
        for indices, row in zip(trained_indices, bank["shadow_membership"]):
            self.assertEqual(len(indices), int(bank["target_training_size"]))
            np.testing.assert_array_equal(
                [index in indices for index in bank["candidate_complete_indices"]], row
            )
        contexts = [call.args[1]["lira_context"] for call in iteration.call_args_list]
        self.assertTrue(all(context is contexts[0] for context in contexts))
        candidate_membership = bank["candidate_membership"]
        candidates = bank["candidate_complete_indices"]
        target_train = set(self.clients[0].indices)
        for indices in contexts[0]["fixed_train_indices"].values():
            target_train.update(indices)
        np.testing.assert_array_equal(
            [index in target_train for index in candidates], candidate_membership
        )

    def test_invalid_shadow_counts_fail_before_any_training(self):
        for count in (-1, 1, 2, 3, 5, 8.0, True):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "num_shadow_models"):
                    self._run_suite(
                        directory, [{"tests": ["LiRA"]}], mock.Mock(), mock.Mock(),
                        dict(self.params, num_shadow_models=count),
                    )
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_config_sweeps_audit_only_three_representative_cases(self):
        for config in (spectral_wip, revised_diagonal):
            with self.subTest(config=config.__name__):
                original = {"retrain_epochs": 1, "tests": ["test_accuracy", "LiRA"]}
                cases = config.generate_params_ranges(original)
                audited = [i for i, case in enumerate(cases) if runner._requests_lira(case)]
                self.assertEqual(audited, [0, len(cases) // 2, len(cases) - 1])
                self.assertTrue(all("test_accuracy" in case["tests"] for case in cases))
                self.assertEqual(original["tests"], ["test_accuracy", "LiRA"])

    def test_zero_reset_reuses_scores_and_unaudited_cases_skip_inference(self):
        instance = runner._RevisedTest.__new__(runner._RevisedTest)
        instance.trained_model = nn.Linear(2, 2)
        instance.shadow_out_model = nn.Linear(2, 2)
        instance.num_total_params = 6
        instance.client_information = {}
        instance.stage_timings = {"score_seconds": 0.0}
        initial = {"trained": {"pred": np.array([0]), "loss": np.array([1.0])}}
        instance.init_eval_test_results = initial
        instance.init_eval_train_results = initial
        instance.lira_candidate_dataset = self.dataset
        instance.init_lira_results = {
            "trained": np.array([0.5, -0.5]), "shadow_out": np.array([0.1, -0.2])
        }
        instance.evaluate_lira_function = mock.Mock(
            side_effect=AssertionError("unexpected repeated LiRA inference")
        )
        case = {"unlearning_method": "information", "unlearning_percentage": 0,
                "retrain_epochs": 1, "tests": ["LiRA"]}
        with mock.patch.object(runner, "find_informative_params", return_value={}):
            instance.run_test(case)
            self.assertEqual(len(instance.last_lira_results), 4)
            self.assertTrue(all(
                score is instance.init_lira_results["trained"]
                for score in instance.last_lira_results.values()
            ))
            instance.run_test(dict(case, tests=[]))
            self.assertIsNone(instance.last_lira_results)
        instance.evaluate_lira_function.assert_not_called()

    def test_sparse_scores_follow_successful_utility_rows(self):
        values = {"trained": {"pred": np.array([0]), "loss": np.array([1.0])}}
        scores = {"reset": np.array([0.3, -0.4])}

        class FakeTest:
            def __init__(self, *args):
                self.init_eval_test_results = values
                self.init_eval_train_results = values

            def configure_lira(self, *args):
                self.init_lira_results = {"trained": scores["reset"]}

            def run_test(self, case):
                if case.get("fail"):
                    raise RuntimeError("intentional failed case")
                self.last_lira_results = scores if runner._requests_lira(case) else None
                return values, values, {"reset_params_percentage": case["percentage"]}

        cases = [
            {"percentage": 0, "tests": ["LiRA"]},
            {"percentage": 10, "fail": True, "tests": ["LiRA"]},
            {"percentage": 20},
            {"percentage": 30, "tests": ["LiRA"]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            args = {
                "test_path": directory, "train_dataset": self.dataset,
                "test_dataset": self.test_dataset, "clients_subsets": self.clients,
                "model_class": lambda: nn.Linear(2, 2), "loss_class": nn.CrossEntropyLoss,
                "trainer_function": None, "init_params_dict": self.params,
                "test_params_dicts": cases, "poisoned_backdoor_dataset": None,
                "clean_backdoor_dataset": None, "save_models": False,
                "lira_context": {"candidate_dataset": self.dataset,
                                 "fixed_train_indices": {}, "fixed_eval_indices": {}},
                "evaluate_lira_function": None,
            }
            errors = runner._run_tests_iter(
                0, args, FakeTest, torch.device("cpu"), filter_error_results=True
            )
            output = Path(directory) / "test_0"
            self.assertEqual(errors, [1])
            self.assertEqual(
                persistence.load_pickle(output, persistence.LIRA_CASE_INDICES), [0, 2]
            )
            self.assertEqual(
                persistence.load_npz(output, persistence.EVAL_LIRA_RESULTS)["reset"].shape,
                (2, 2),
            )
            self.assertEqual(
                persistence.load_npz(output, persistence.EVAL_TEST_RESULTS)["trained__pred"].shape,
                (3, 1),
            )


if __name__ == "__main__":
    unittest.main()
