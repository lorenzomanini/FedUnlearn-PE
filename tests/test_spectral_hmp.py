"""Numerical regression checks for batching BackPACK direction vectors."""

import copy
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Load the estimator alone so these checks do not need the privacy/plotting stack.
_path = Path(__file__).resolve().parents[1] / "fisherunlearn/information/spectral_wip.py"
_spec = importlib.util.spec_from_file_location("spectral_wip", _path)
spectral = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spectral)


class SpectralHMPTests(unittest.TestCase):
    def test_chunked_convolution_matches_batched_product(self):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8, 3)
        ).double().eval()
        # Include an incomplete data batch and an incomplete vector chunk.
        dataset = TensorDataset(
            torch.randn(5, 1, 4, 4, dtype=torch.float64),
            torch.tensor([0, 1, 2, 1, 0]),
        )
        loader = DataLoader(dataset, batch_size=3)
        n_params = sum(p.numel() for p in model.parameters())
        directions = torch.randn(n_params, 3, dtype=torch.float64)

        unchunked, *_ = spectral.make_block_hessian_matvec(
            copy.deepcopy(model), loader, nn.CrossEntropyLoss(), "cpu", hmp_chunk_size=3
        )
        expected = unchunked(directions)
        self.assertTrue(torch.isfinite(expected).all())
        for chunk_size in (1, 2):
            with self.subTest(chunk_size=chunk_size):
                chunked, *_ = spectral.make_block_hessian_matvec(
                    copy.deepcopy(model), loader, nn.CrossEntropyLoss(), "cpu",
                    hmp_chunk_size=chunk_size,
                )
                torch.testing.assert_close(
                    chunked(directions), expected, rtol=1e-9, atol=1e-11
                )

    def test_rejects_invalid_chunk_sizes(self):
        for chunk_size in (0, -1, 1.5, True):
            with self.subTest(chunk_size=chunk_size), self.assertRaises(ValueError):
                spectral.make_block_hessian_matvec(
                    None, None, None, "cpu", hmp_chunk_size=chunk_size
                )


if __name__ == "__main__":
    unittest.main()
