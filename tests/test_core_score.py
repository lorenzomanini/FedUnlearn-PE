"""Small exact checks of the noncommuting curvature core and full HVP path."""

import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


_path = Path(__file__).resolve().parents[1] / "fisherunlearn/information/spectral_wip.py"
_spec = importlib.util.spec_from_file_location("spectral_core", _path)
spectral = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spectral)


class CoreScoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.dtype = torch.float64

    def test_noncommuting_target_keeps_off_diagonal_information(self):
        basis = torch.eye(2, dtype=self.dtype)
        full = torch.tensor([4., 1.], dtype=self.dtype)
        target = torch.tensor([[2., 1.], [1., 3.]], dtype=self.dtype)
        result = spectral.core_score_from_subspace(basis, full, target)
        # B=[[.5,.5],[.5,3]]; the old diagonal-only scores were [.25,9].
        torch.testing.assert_close(result["diag_flat"], torch.tensor([.5, 9.25], dtype=self.dtype))
        self.assertAlmostEqual(result["diagnostics"]["score_sum"], 9.75)
        self.assertLess(result["diagnostics"]["sum_rule_absolute_error"], 1e-12)

    def test_diagonal_reduces_to_squared_curvature_ratios(self):
        full = torch.tensor([2., 4., 8.], dtype=self.dtype)
        target = torch.diag(torch.tensor([3., 2., -4.], dtype=self.dtype))
        result = spectral.core_score_from_subspace(torch.eye(3, dtype=self.dtype), full, target)
        torch.testing.assert_close(result["diag_flat"], (target.diagonal() / full).square())

    def test_invariant_to_orthogonal_subspace_basis_rotation(self):
        basis, _ = torch.linalg.qr(torch.randn(7, 3, dtype=self.dtype))
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=self.dtype))
        full = torch.diag(torch.tensor([1., 3., 7.], dtype=self.dtype))
        target = torch.randn(3, 3, dtype=self.dtype)
        target = target + target.T
        original = spectral.core_score_from_subspace(basis, full, target)
        rotated = spectral.core_score_from_subspace(
            basis @ rotation, rotation.T @ full @ rotation, rotation.T @ target @ rotation,
        )
        torch.testing.assert_close(original["diag_flat"], rotated["diag_flat"], atol=1e-12, rtol=1e-12)
        self.assertLess(rotated["diagnostics"]["sum_rule_relative_error"], 1e-12)

    def test_positive_support_excludes_negative_zero_and_threshold_boundary(self):
        full = torch.tensor([-3., 0., 1e-5, 1e-3, 1.], dtype=self.dtype)
        result = spectral.core_score_from_subspace(
            torch.eye(5, dtype=self.dtype), full, torch.ones(5, 5, dtype=self.dtype),
            eigenvalue_threshold=1e-5, relative_eigenvalue_threshold=1e-3,
        )
        self.assertEqual(result["diagnostics"]["retained_rank"], 1)
        torch.testing.assert_close(result["diag_flat"], torch.tensor([0., 0., 0., 0., 1.], dtype=self.dtype))
        self.assertEqual(result["diagnostics"]["eigenvalue_threshold"], 1e-3)

    def test_no_positive_curvature_returns_zero_information(self):
        result = spectral.core_score_from_subspace(
            torch.eye(2, dtype=self.dtype), torch.tensor([-1., 0.], dtype=self.dtype),
            torch.ones(2, 2, dtype=self.dtype),
        )
        torch.testing.assert_close(result["diag_flat"], torch.zeros(2, dtype=self.dtype))
        self.assertEqual(result["diagnostics"]["retained_rank"], 0)
        self.assertEqual(result["diagnostics"]["sum_rule_relative_error"], 0.)

    def test_full_hvp_matches_dense_hessian_including_cross_parameter_blocks(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1)).double().eval()
        x, y = torch.randn(5, 2, dtype=self.dtype), torch.randn(5, 1, dtype=self.dtype)
        loader = DataLoader(TensorDataset(x, y), batch_size=3)
        names = [name for name, _ in model.named_parameters()]
        params = list(model.parameters())
        shapes, numels, size = spectral.get_param_info(params)
        flat = spectral.flatten_params(params)

        def objective(point):
            values = spectral.unflatten_like(point, shapes, numels)
            predicted = torch.func.functional_call(model, dict(zip(names, values)), (x,))
            return nn.functional.mse_loss(predicted, y)

        dense = torch.autograd.functional.hessian(objective, flat)
        # Cross-tensor entries exist, so a block approximation cannot pass.
        self.assertGreater(dense[:numels[0], numels[0]:].abs().sum().item(), .01)
        directions = torch.randn(size, 3, dtype=self.dtype)
        for chunk in (1, 2, 3):
            with self.subTest(chunk=chunk):
                operator, *_ = spectral.make_hessian_matvec(
                    model, loader, nn.MSELoss(), "cpu", hmp_chunk_size=chunk,
                )
                torch.testing.assert_close(operator(directions), dense @ directions, atol=1e-11, rtol=1e-10)
                torch.testing.assert_close(operator(directions), dense @ directions, atol=1e-11, rtol=1e-10)

    def test_full_operator_preserves_modes_weights_and_existing_gradients(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(), nn.Linear(2, 1)).double().train()
        model[0].eval()
        loader = DataLoader(TensorDataset(torch.randn(3, 2, dtype=self.dtype),
                                         torch.randn(3, 1, dtype=self.dtype)), batch_size=2)
        modes = [m.training for m in model.modules()]
        params = list(model.parameters())
        original = spectral.flatten_params(params).clone()
        for p in params:
            p.grad = torch.ones_like(p)
        operator, *_ = spectral.make_hessian_matvec(model, loader, nn.MSELoss(), "cpu")
        direction = torch.randn(original.numel(), 1, dtype=self.dtype)
        first, second = operator(direction), operator(direction)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(spectral.flatten_params(params), original)
        self.assertEqual(modes, [m.training for m in model.modules()])
        for p in params:
            torch.testing.assert_close(p.grad, torch.ones_like(p))

    def test_end_to_end_exact_quadratic_and_legacy_alias(self):
        model = nn.Linear(2, 1, bias=False).double().eval()
        x = torch.tensor([[2., 0.], [0., 1.]], dtype=self.dtype)
        target_x = torch.tensor([[1., 1.], [0., 1.]], dtype=self.dtype)
        y = torch.zeros(2, 1, dtype=self.dtype)
        loader = DataLoader(TensorDataset(x, y), batch_size=1)
        target_loader = DataLoader(TensorDataset(target_x, y), batch_size=1)
        # H=diag(4,1), H_T=[[1,1],[1,2]], B=[[.25,.5],[.5,2]].
        expected = torch.tensor([.3125, 4.25], dtype=self.dtype)
        for estimator in (spectral.estimate_core_score, spectral.estimate_diag_commuting_backpack):
            with self.subTest(estimator=estimator.__name__):
                result = estimator(model, loader, target_loader, nn.MSELoss(), 2, "cpu", num_power_iters=0)
                torch.testing.assert_close(result["diag_flat"], expected, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(result["diag_by_name"]["weight"], expected.reshape(1, 2))
                self.assertTrue(result["diagnostics"]["subspace"]["converged"])
                self.assertEqual(result["diagnostics"]["subspace"]["operator_calls"], 1)
                self.assertEqual(result["diagnostics"]["target_hvp_directions"], 2)

    def test_score_agreement_measures_log_correlation_and_top_overlap(self):
        scores = torch.tensor([2.0 ** power for power in range(10)], dtype=self.dtype)
        scaled = spectral.score_agreement(scores, 3 * scores)
        self.assertAlmostEqual(scaled["log_score_correlation"], 1.0)
        self.assertEqual(scaled["top_set_overlap"], 1.0)
        reversed_scores = spectral.score_agreement(scores, scores.flip(0))
        self.assertAlmostEqual(reversed_scores["log_score_correlation"], -1.0)
        self.assertEqual(reversed_scores["top_set_overlap"], 0.0)

    def test_control_sample_is_scored_in_the_target_basis(self):
        model = nn.Linear(2, 1, bias=False).double().eval()
        x = torch.tensor([[2., 0.], [0., 1.]], dtype=self.dtype)
        target_x = torch.tensor([[1., 1.], [0., 1.]], dtype=self.dtype)
        y = torch.zeros(2, 1, dtype=self.dtype)
        loader = DataLoader(TensorDataset(x, y), batch_size=1)
        target_loader = DataLoader(TensorDataset(target_x, y), batch_size=1)
        score = lambda control: spectral.estimate_core_score(
            model, loader, target_loader, nn.MSELoss(), 2, "cpu", num_power_iters=0,
            loader_control=control,
        )
        self.assertNotIn("control", score(None)["diagnostics"])
        same = score(target_loader)
        control = same["diagnostics"]["control"]
        torch.testing.assert_close(control["whitened_target"], same["whitened_target"])
        self.assertAlmostEqual(control["log_score_correlation"], 1.0)
        self.assertEqual(control["top_set_overlap"], 1.0)
        # Rescoring the full-data sample whitens it to the identity.
        torch.testing.assert_close(score(loader)["diagnostics"]["control"]["whitened_target"],
                                   torch.eye(2, dtype=self.dtype))

    def test_gauss_newton_products_match_dense_and_are_positive_semidefinite(self):
        torch.manual_seed(5)
        model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 3)).double()
        names = [name for name, _ in model.named_parameters()]
        shapes = [value.shape for value in model.parameters()]
        theta = torch.cat([value.detach().reshape(-1) for value in model.parameters()])

        def unflatten(vector):
            values, offset = {}, 0
            for name, shape in zip(names, shapes):
                values[name] = vector[offset:offset + shape.numel()].reshape(shape)
                offset += shape.numel()
            return values

        for loss_fn, labels in ((nn.CrossEntropyLoss(), torch.randint(3, (5,))),
                                (nn.MSELoss(), torch.randn(5, 3, dtype=self.dtype))):
            loader = DataLoader(TensorDataset(torch.randn(5, 3, dtype=self.dtype), labels),
                                batch_size=3)  # uneven batches of 3 and 2
            dense = torch.zeros(len(theta), len(theta), dtype=self.dtype)
            for x, y in loader:
                forward = lambda vector: torch.func.functional_call(model, unflatten(vector), (x,))
                outputs = forward(theta)
                jacobian = torch.autograd.functional.jacobian(forward, theta).reshape(outputs.numel(), -1)
                curvature = torch.autograd.functional.hessian(
                    lambda flat: loss_fn(flat.reshape(outputs.shape), y), outputs.reshape(-1))
                dense += len(x) * jacobian.T @ curvature @ jacobian / 5
            for chunk in (1, 2, 3):
                with self.subTest(loss=type(loss_fn).__name__, chunk=chunk):
                    operator, *_ = spectral.make_ggn_matvec(model, loader, loss_fn, "cpu",
                                                            hmp_chunk_size=chunk)
                    products = operator(torch.eye(len(theta), dtype=self.dtype))
                    torch.testing.assert_close(products, dense, atol=1e-12, rtol=1e-12)
                    self.assertGreater(torch.linalg.eigvalsh(products).min().item(), -1e-12)

    def test_generalized_eigenspace_matches_dense_pencil(self):
        import numpy as np
        import scipy.linalg
        q, _ = torch.linalg.qr(torch.randn(200, 200, dtype=self.dtype))
        outer = torch.randn(30, 30, dtype=self.dtype)
        metric = torch.randn(30, 30, dtype=self.dtype)
        mass = torch.diag(torch.rand(200, dtype=self.dtype) + 0.5)
        cases = {
            "generic": (outer.T @ outer, metric.T @ metric + 0.1 * torch.eye(30, dtype=self.dtype)),
            # Three outliers above a tight cluster, like a target-specific signature.
            "clustered": (torch.linalg.cholesky(q @ mass @ q.T) @ q @ torch.diag(torch.cat([
                torch.tensor([9., 7., 6.5], dtype=self.dtype),
                1 + 1e-3 * torch.randn(197, dtype=self.dtype)])) @ q.T
                @ torch.linalg.cholesky(q @ mass @ q.T).T, q @ mass @ q.T),
        }
        for name, (a, b) in cases.items():
            with self.subTest(case=name):
                values, vectors = spectral.top_generalized_eigenspace(
                    lambda block: (a @ block, b @ block), len(a), 3, "cpu", self.dtype,
                    num_iters=300, tolerance=1e-10)
                reference, eigenvectors = scipy.linalg.eigh(a.numpy(), b.numpy())
                np.testing.assert_allclose(values.numpy(), reference[::-1][:3], rtol=1e-8)
                torch.testing.assert_close(vectors.T @ b @ vectors, torch.eye(3, dtype=self.dtype),
                                           atol=1e-8, rtol=0)
                span, _ = np.linalg.qr(vectors.numpy())
                expected, _ = np.linalg.qr(eigenvectors[:, ::-1][:, :3])
                cosines = np.linalg.svd(span.T @ expected, compute_uv=False)
                self.assertGreater(cosines.min(), 1 - 1e-8)

    def test_generalized_core_selects_the_target_dominated_direction(self):
        model = nn.Linear(2, 1, bias=False).double().eval()
        y = torch.zeros(2, 1, dtype=self.dtype)
        loader = lambda rows: DataLoader(TensorDataset(torch.tensor(rows, dtype=self.dtype), y), batch_size=1)
        retained, target = loader([[10., 0.], [0., 1.]]), loader([[1., 0.], [0., 3.]])
        # H_R = diag(100, 1), H_T = diag(1, 9) and a = 0.1: H = diag(90.1, 1.8), so the
        # relative curvatures are (0.011, 5) under the bound 1/a = 10.
        result = spectral.estimate_generalized_core_score(
            model, retained, target, nn.MSELoss(), 1, "cpu", 0.1, num_iters=20,
            loader_control=loader([[3., 0.], [0., 1.]]),
        )
        torch.testing.assert_close(result["diag_flat"], torch.tensor([0., 25.], dtype=self.dtype))
        torch.testing.assert_close(result["whitened_target"], torch.tensor([[5.]], dtype=self.dtype))
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["relative_curvature_bound"], 10.0)
        self.assertTrue(diagnostics["subspace"]["converged"])
        # Damping can only lower the relative curvature.
        self.assertLess(diagnostics["generalized_eigenvalues"][0], 5.0)
        self.assertGreater(diagnostics["generalized_eigenvalues"][0], 4.9)
        torch.testing.assert_close(diagnostics["control"]["whitened_target"],
                                   torch.tensor([[1 / 1.8]], dtype=self.dtype))
        # The top-curvature subspace of the same H keeps the shared direction instead.
        both = DataLoader(TensorDataset(torch.tensor([[10., 0.], [0., 1.], [1., 0.], [0., 3.]],
                                                     dtype=self.dtype), torch.zeros(4, 1, dtype=self.dtype)),
                          batch_size=2)
        top = spectral.estimate_core_score(model, both, target, nn.MSELoss(), 1, "cpu",
                                           num_power_iters=20, curvature_backend="ggn")
        self.assertLess(top["whitened_target"].item(), 0.1)
        self.assertGreater(top["diag_flat"][0], top["diag_flat"][1])

    def test_generalized_core_needs_gauss_newton_and_a_valid_fraction(self):
        model = nn.Linear(2, 1, bias=False).double().eval()
        loader = DataLoader(TensorDataset(torch.eye(2, dtype=self.dtype), torch.zeros(2, 1, dtype=self.dtype)))
        for options in ({"curvature_backend": "full"}, {"target_fraction": 0.0},
                        {"target_fraction": 1.0}, {"damping": -1.0}):
            arguments = {"target_fraction": 0.5, **options}
            with self.subTest(**options), self.assertRaises(ValueError):
                spectral.estimate_generalized_core_score(model, loader, loader, nn.MSELoss(), 1, "cpu",
                                                         num_iters=1, **arguments)

    def test_empty_loader_and_invalid_threshold_fail_clearly(self):
        model = nn.Linear(2, 1).double().eval()
        loader = DataLoader(TensorDataset(torch.empty(0, 2, dtype=self.dtype),
                                         torch.empty(0, 1, dtype=self.dtype)), batch_size=2)
        operator, *_ = spectral.make_hessian_matvec(model, loader, nn.MSELoss(), "cpu")
        with self.assertRaisesRegex(ValueError, "empty loader"):
            operator(torch.ones(3, 1, dtype=self.dtype))
        for value in (-1., float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                spectral.core_score_from_subspace(torch.eye(2, dtype=self.dtype),
                                                 torch.ones(2, dtype=self.dtype),
                                                 torch.eye(2, dtype=self.dtype),
                                                 eigenvalue_threshold=value)

    def test_subspace_stops_when_residual_has_converged(self):
        diagonal = torch.tensor([5., 2., 0., 0.], dtype=self.dtype)
        values, vectors, diagnostics = spectral.top_eigenspace_block_power(
            lambda directions: diagonal[:, None] * directions,
            4, 2, "cpu", self.dtype, num_iters=10,
            tolerance=1e-10, return_diagnostics=True,
        )
        torch.testing.assert_close(values, diagonal[:2])
        self.assertTrue(diagnostics["converged"])
        self.assertEqual(diagnostics["operator_calls"], 2)
        torch.testing.assert_close(diagonal[:, None] * vectors, vectors * values,
                                   atol=1e-12, rtol=1e-12)

    def test_constant_gradient_has_zero_hessian(self):
        class LinearLoss(nn.Module):
            def forward(self, prediction, target):
                return prediction.mean()

        model = nn.Linear(2, 1, bias=False).double()
        loader = DataLoader(TensorDataset(torch.randn(3, 2, dtype=self.dtype),
                                         torch.zeros(3, 1, dtype=self.dtype)), batch_size=2)
        operator, *_ = spectral.make_hessian_matvec(model, loader, LinearLoss(), "cpu")
        torch.testing.assert_close(operator(torch.eye(2, dtype=self.dtype)),
                                   torch.zeros(2, 2, dtype=self.dtype))


if __name__ == "__main__":
    unittest.main()
