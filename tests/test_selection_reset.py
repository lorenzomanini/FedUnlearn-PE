"""Score-mass selection and selective retraining regression checks."""

import copy
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


def _load_module(name, relative_path):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selection = _load_module("selection", "fisherunlearn/information/selection.py")
unlearning = _load_module("unlearning", "fisherunlearn/unlearning.py")


class ScoreSelectionTests(unittest.TestCase):
    def select(self, values, percentage, method="information", **kwargs):
        return selection.find_informative_params(
            {"weight": torch.as_tensor(values)}, method, percentage, **kwargs
        )["weight"]

    def test_smallest_prefix_reaches_requested_mass(self):
        values = torch.tensor([4.0, 3.0, 2.0, 1.0])
        for percentage, expected_count in [(0, 0), (1, 1), (40, 1), (41, 2), (80, 3), (100, 4)]:
            with self.subTest(percentage=percentage):
                indices = self.select(values, percentage)
                self.assertEqual(len(indices), expected_count)
                if expected_count:
                    mass = values[indices[:, 0]].sum()
                    self.assertGreaterEqual(mass.item(), values.sum().item() * percentage / 100)
                    self.assertLess((mass - values[indices[-1, 0]]).item(), values.sum().item() * percentage / 100)

    def test_ties_choose_exact_deterministic_count(self):
        expected = torch.tensor([[0, 0], [0, 1]])
        for method in ("information", "parameters"):
            torch.testing.assert_close(self.select(torch.ones(2, 2), 50, method), expected)

    def test_zero_group_and_full_mass_with_zero_scores(self):
        self.assertEqual(len(self.select(torch.zeros(4), 100)), 0)
        self.assertEqual(len(self.select(torch.empty(0), 100)), 0)
        torch.testing.assert_close(self.select([3.0, 0.0, 0.0], 100), torch.tensor([[0]]))

    def test_parameter_and_random_boundaries(self):
        for method in ("parameters", "random"):
            for percentage, count in ((0, 0), (50, 2), (100, 5)):
                indices = self.select(torch.ones(5), percentage, method)
                self.assertEqual(len(indices), count)
                self.assertEqual(indices.unique().numel(), count)

    def test_filters_and_tuple_output(self):
        information = {name: torch.ones(2, 2) for name in ("a", "b", "c")}
        result = selection.find_informative_params(
            information, "information", 50,
            whitelist=["a", "b"], blacklist=["b"], tuple_out=True,
        )
        self.assertEqual(list(result), ["a"])
        torch.testing.assert_close(information["a"][result["a"]], torch.ones(2))

    def test_invalid_scores_and_percentage_fail_clearly(self):
        for values in ([1.0, -1.0], [1.0, float("nan")], [float("inf")]):
            with self.assertRaisesRegex(ValueError, "scores"):
                self.select(values, 50)
        for percentage in (-1, 101, float("nan")):
            with self.assertRaisesRegex(ValueError, "percentage"):
                self.select([1.0], percentage)

    def test_tensor_with_gradients_and_large_scores(self):
        values = torch.tensor([3e38, 2e38, 1e38], requires_grad=True)
        torch.testing.assert_close(self.select(values, 60), torch.tensor([[0], [1]]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_selection_stays_on_cuda(self):
        for method in ("information", "parameters", "random"):
            indices = self.select(torch.ones(4, device="cuda"), 50, method)
            self.assertEqual(indices.device.type, "cuda")
            self.assertEqual(len(indices), 2)


class NamedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # These would collide if dots were simply replaced with underscores.
        self.block_a = nn.Linear(3, 2)
        self.block = nn.Module()
        self.block.a_weight = nn.Parameter(torch.ones(2, 3))

    def forward(self, inputs, scale=1):
        return scale * (self.block_a(inputs) + inputs @ self.block.a_weight.t())


class SelectiveResetTests(unittest.TestCase):
    def test_reference_reset_preserves_unselected_values_and_batchnorm_buffers(self):
        model = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3)).double()
        reference = copy.deepcopy(model)
        with torch.no_grad():
            model[1].running_mean.fill_(7)
            reference[0].weight.fill_(2)
            reference[1].weight.fill_(1)
            reference[1].running_mean.fill_(-8)
        before = copy.deepcopy(model.state_dict())
        reference_before = copy.deepcopy(reference.state_dict())
        indices = {"0.weight": torch.tensor([[0, 1]]), "1.weight": torch.tensor([[2]])}
        reset = unlearning.reset_parameters(model, indices, reference)
        wrapper = unlearning.UnlearnNet(model, indices, reference.state_dict())
        for name, original in before.items():
            expected = original.clone()
            if name in indices:
                expected[tuple(indices[name].t())] = reference_before[name][tuple(indices[name].t())]
            torch.testing.assert_close(reset[name], expected, rtol=0, atol=0)
            torch.testing.assert_close(wrapper.get_retrained_params()[name], expected, rtol=0, atol=0)
            torch.testing.assert_close(model.state_dict()[name], original, rtol=0, atol=0)
            torch.testing.assert_close(reference.state_dict()[name], reference_before[name], rtol=0, atol=0)

    def test_reference_initialized_updates_match_dense_mask_reference(self):
        torch.manual_seed(29)
        model = nn.Sequential(nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 2)).double()
        initialization = nn.Sequential(nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 2)).double()
        indices = {"0.weight": torch.tensor([[0, 1], [2, 0]]), "2.bias": torch.tensor([[1]])}
        wrapper = unlearning.UnlearnNet(model, indices, initialization)
        dense = copy.deepcopy(model)
        dense.load_state_dict(unlearning.reset_parameters(model, indices, initialization))
        inputs, targets = torch.randn(8, 3, dtype=torch.float64), torch.randn(8, 2, dtype=torch.float64)
        optimizers = [torch.optim.SGD(m.parameters(), lr=.1) for m in (wrapper, dense)]
        for _ in range(3):
            for optimizer in optimizers:
                optimizer.zero_grad()
            torch.testing.assert_close(wrapper(inputs), dense(inputs))
            for net in (wrapper, dense):
                nn.functional.mse_loss(net(inputs), targets).backward()
            for name, parameter in dense.named_parameters():
                mask = torch.zeros_like(parameter)
                if name in indices:
                    mask[tuple(indices[name].t())] = 1
                parameter.grad.mul_(mask)
            for optimizer in optimizers:
                optimizer.step()
        for name, value in wrapper.get_retrained_params().items():
            torch.testing.assert_close(value, dense.state_dict()[name])

    def test_reference_reset_reopens_dead_batchnorm_relu_channel(self):
        torch.manual_seed(7)
        model = nn.Sequential(
            nn.Conv2d(1, 2, 3, padding=1, bias=False), nn.BatchNorm2d(2),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2, 2),
        )
        indices = {"1.weight": torch.tensor([[0]]), "1.bias": torch.tensor([[0]])}
        inputs, targets = torch.randn(8, 1, 4, 4), torch.arange(8) % 2
        zero = unlearning.UnlearnNet(model, indices)
        initialized = unlearning.UnlearnNet(model, indices, copy.deepcopy(model))
        for wrapper in (zero, initialized):
            nn.functional.cross_entropy(wrapper(inputs), targets).backward()
        for parameter in zero.parameters():
            self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in initialized.parameters()), 0)

    def test_invalid_reference_shape_and_buffer_selection_are_rejected(self):
        model = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
        indices = {"0.weight": torch.tensor([[0, 1]])}
        for reference in ({}, {"0.weight": torch.zeros(1)}):
            with self.assertRaisesRegex(ValueError, "shape-compatible"):
                unlearning.reset_parameters(model, indices, reference)
        with self.assertRaisesRegex(ValueError, "only model parameters"):
            unlearning.reset_parameters(
                model, {"1.running_mean": torch.tensor([[0]])}, model.state_dict(),
            )

    def test_reset_matches_coordinates_and_does_not_mutate_input(self):
        model = nn.Linear(3, 2).double()
        before = copy.deepcopy(model.state_dict())
        coordinates = torch.tensor([[0, 1], [1, 2]])
        reset = unlearning.reset_parameters(model, {"weight": coordinates})
        expected = before["weight"].clone()
        expected[coordinates[:, 0], coordinates[:, 1]] = 0
        torch.testing.assert_close(reset["weight"], expected)
        torch.testing.assert_close(reset["bias"], before["bias"])
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, before[name])
        tuple_reset = unlearning.reset_parameters(model, {"weight": tuple(coordinates.t())})
        torch.testing.assert_close(tuple_reset["weight"], expected)

    def test_updates_and_gradients_match_dense_mask_reference(self):
        torch.manual_seed(17)
        model = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2)).double()
        indices = {"0.weight": torch.tensor([[0, 1], [2, 3]]), "2.bias": torch.tensor([[1]])}
        wrapper = unlearning.UnlearnNet(model, indices)
        reference = copy.deepcopy(model)
        reference.load_state_dict(unlearning.reset_parameters(model, indices))
        initial = copy.deepcopy(reference.state_dict())
        inputs = torch.randn(8, 4, dtype=torch.float64)
        targets = torch.randn(8, 2, dtype=torch.float64)
        optimizer = torch.optim.SGD(wrapper.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        self.assertEqual(sum(p.numel() for p in wrapper.parameters()), 3)
        self.assertTrue(all(p.dtype == torch.float64 for p in wrapper.parameters()))
        self.assertTrue(all(not p.requires_grad for p in wrapper.inner_model["model"].parameters()))
        for _ in range(3):
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            outputs = wrapper(inputs)
            expected_outputs = reference(inputs)
            torch.testing.assert_close(outputs, expected_outputs)
            ((outputs - targets) ** 2).mean().backward()
            ((expected_outputs - targets) ** 2).mean().backward()
            for name, parameter in reference.named_parameters():
                mask = torch.zeros_like(parameter)
                if name in indices:
                    mask[tuple(indices[name].t())] = 1
                    key = wrapper._selected_names[name]
                    torch.testing.assert_close(wrapper.retrain_params[key].grad, parameter.grad[tuple(indices[name].t())])
                parameter.grad.mul_(mask)
            optimizer.step()
            reference_optimizer.step()
        final = wrapper.get_retrained_params()
        for name, expected in reference.state_dict().items():
            torch.testing.assert_close(final[name], expected)
            mask = torch.ones_like(expected, dtype=torch.bool)
            if name in indices:
                mask[tuple(indices[name].t())] = False
            torch.testing.assert_close(final[name][mask], initial[name][mask], rtol=0, atol=0)
        self.assertTrue(all(tensor.layout == torch.strided for tensor in wrapper.state_dict().values()))

    def test_original_names_and_state_roundtrip(self):
        model = NamedModel()
        indices = {
            "block_a.weight": torch.tensor([[0, 1]]),
            "block.a_weight": torch.tensor([[1, 2]]),
        }
        wrapper = unlearning.UnlearnNet(model, indices)
        with torch.no_grad():
            for index, parameter in enumerate(wrapper.parameters()):
                parameter.fill_(index + 2)
        rebuilt = NamedModel()
        rebuilt.load_state_dict(wrapper.get_retrained_params())
        inputs = torch.randn(4, 3)
        torch.testing.assert_close(wrapper(inputs, scale=2), rebuilt(inputs, scale=2))
        loaded = unlearning.UnlearnNet(model, indices)
        loaded.load_state_dict(wrapper.state_dict())
        torch.testing.assert_close(loaded(inputs), wrapper(inputs))
        torch.testing.assert_close(copy.deepcopy(wrapper)(inputs), wrapper(inputs))

    def test_eval_and_train_control_batchnorm_buffers(self):
        model = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.Dropout(0.2))
        wrapper = unlearning.UnlearnNet(model, {"0.weight": torch.tensor([[0, 1]])})
        inputs = torch.randn(8, 3)
        wrapper.eval()
        before = wrapper.get_retrained_params()
        first = wrapper(inputs)
        second = wrapper(inputs)
        torch.testing.assert_close(first, second)
        self.assertFalse(wrapper.inner_model["model"].training)
        torch.testing.assert_close(wrapper.get_retrained_params()["1.running_mean"], before["1.running_mean"])
        wrapper.train()
        wrapper(inputs)
        self.assertTrue(wrapper.inner_model["model"].training)
        self.assertEqual(wrapper.get_retrained_params()["1.num_batches_tracked"].item(), 1)

    def test_device_and_dtype_moves_preserve_selected_values(self):
        wrapper = unlearning.UnlearnNet(nn.Linear(3, 2), {"weight": torch.tensor([[0, 1]])}).double()
        self.assertEqual(wrapper(torch.randn(4, 3, dtype=torch.float64)).dtype, torch.float64)
        self.assertEqual(wrapper.indices_weight.dtype, torch.long)
        if torch.cuda.is_available():
            wrapper.cuda()
            self.assertEqual(wrapper(torch.randn(4, 3, dtype=torch.float64, device="cuda")).device.type, "cuda")

    def test_empty_selection_preserves_model(self):
        model = nn.Linear(3, 2).eval()
        wrapper = unlearning.UnlearnNet(model, {"weight": torch.empty((0, 2), dtype=torch.long)})
        self.assertEqual(list(wrapper.parameters()), [])
        inputs = torch.randn(4, 3)
        torch.testing.assert_close(wrapper(inputs), model(inputs))
        self.assertFalse(wrapper.training)

    def test_duplicate_and_out_of_bounds_coordinates_are_rejected(self):
        model = nn.Linear(3, 2)
        for coordinates in (torch.tensor([[0, 1], [0, 1]]), torch.tensor([[0, 3]])):
            with self.assertRaises(ValueError):
                unlearning.UnlearnNet(model, {"weight": coordinates})


if __name__ == "__main__":
    unittest.main()
